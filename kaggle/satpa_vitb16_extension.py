"""Kaggle notebook source: locked SA-TPA ViT-B/16 extension."""

import json
import os
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

gpu_names = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
)
if "P100" in gpu_names:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--index-url",
        "https://download.pytorch.org/whl/cu124", "torch==2.6.0", "torchvision==0.21.0",
    ])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch==2.26.1"])

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

import open_clip


WORK = Path("/kaggle/working/satpa_vitb16")
WORK.mkdir(parents=True, exist_ok=True)
PROMPTS = (
    "a photo of a {}.", "a product photo of a {}.",
    "an image of a {}.", "a close-up photo of a {}.",
)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


def find_domain_parent(domain_names):
    input_root = Path("/kaggle/input")
    for first in input_root.rglob(domain_names[0]):
        if first.is_dir() and all((first.parent / name).is_dir() for name in domain_names):
            return first.parent
    raise FileNotFoundError(f"Could not locate domains {domain_names} below {input_root}")


def find_normalized_domains(domain_names, input_root=Path("/kaggle/input")):
    wanted = {re.sub(r"[^a-z0-9]", "", name.lower()): name for name in domain_names}
    for art in input_root.rglob("*"):
        if not art.is_dir() or re.sub(r"[^a-z0-9]", "", art.name.lower()) != "art":
            continue
        children = {
            re.sub(r"[^a-z0-9]", "", child.name.lower()): child
            for child in art.parent.iterdir() if child.is_dir()
        }
        if set(wanted).issubset(children):
            return {original: children[key] for key, original in wanted.items()}
    raise FileNotFoundError(f"Could not locate normalized domains {domain_names}")


def download_officehome_fallback():
    archive = WORK / "officehome.zip"
    extracted = WORK / "officehome_download"
    if not extracted.exists():
        print("Office-Home mount missing; downloading public Kaggle dataset fallback")
        urllib.request.urlretrieve(
            "https://www.kaggle.com/api/v1/datasets/download/ziyankhanpathan/officehomedataset",
            archive,
        )
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extracted)
    return extracted


def resolve_jobs():
    office31 = find_domain_parent(("amazon", "dslr", "webcam"))
    try:
        officehome = find_normalized_domains(("Art", "Clipart", "Product", "Real World"))
    except FileNotFoundError:
        officehome = find_normalized_domains(
            ("Art", "Clipart", "Product", "Real World"), download_officehome_fallback()
        )
    print("Office-31 root:", office31)
    print("Office-Home domains:", officehome)
    return [
        ("office31_amazon", str(office31 / "amazon")),
        ("office31_dslr", str(office31 / "dslr")),
        ("office31_webcam", str(office31 / "webcam")),
        ("officehome_art", str(officehome["Art"])),
        ("officehome_clipart", str(officehome["Clipart"])),
        ("officehome_product", str(officehome["Product"])),
        ("officehome_real_world", str(officehome["Real World"])),
    ]


class DomainDataset(Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.class_to_index = {name: i for i, name in enumerate(self.classes)}
        self.paths = sorted(p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        self.labels = np.asarray([self.class_to_index[p.parent.name] for p in self.paths], dtype=np.int16)
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def readable(name):
    return {"back_pack": "backpack", "bike": "bicycle", "phone": "telephone"}.get(
        name, name.replace("_", " ")
    )


def extract_worker(rank, world_size, assignments):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(device).eval()
    for name, root in assignments[rank]:
        dataset = DomainDataset(root, preprocess)
        per_prompt, ensemble = [], []
        with torch.no_grad():
            for class_name in dataset.classes:
                tokens = tokenizer([p.format(readable(class_name)) for p in PROMPTS]).to(device)
                features = model.encode_text(tokens)
                features = features / features.norm(dim=1, keepdim=True)
                prototype = features.mean(0)
                per_prompt.append(features.cpu())
                ensemble.append((prototype / prototype.norm()).cpu())
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
        images = []
        with torch.no_grad():
            for batch in loader:
                features = model.encode_image(batch.to(device, non_blocking=True))
                images.append((features / features.norm(dim=1, keepdim=True)).cpu().numpy().astype(np.float16))
        np.savez_compressed(
            WORK / f"{name}_vitb16_openai.npz",
            image_features=np.concatenate(images),
            text_features=torch.stack(ensemble).numpy().astype(np.float16),
            text_features_per_prompt=torch.stack(per_prompt).numpy().astype(np.float16),
            classes=np.asarray(dataset.classes), sample_ids=np.arange(len(dataset), dtype=np.int32),
        )
        np.savez_compressed(
            WORK / f"{name}_vitb16_openai.source_labels.npz",
            sample_ids=np.arange(len(dataset), dtype=np.int32), labels=dataset.labels,
        )


def normalize(x, axis=1):
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12, None)


def softmax(x):
    x = x.astype(np.float64); x -= x.max(1, keepdims=True); e = np.exp(x)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def classify(x, p): return softmax(100.0 * normalize(x) @ normalize(p).T)


def source_proto(x, y, text):
    result = []
    for c in range(len(text)):
        z = x[y == c]
        result.append(text[c] if len(z) == 0 else normalize(z.mean(0, keepdims=True))[0])
    return normalize(np.stack(result))


def target_proto(x, probabilities, fallback):
    frequency = np.clip(probabilities.mean(0), 1e-8, None)
    corrected = probabilities / frequency[None, :] ** 0.1
    corrected /= corrected.sum(1, keepdims=True)
    ordered = np.sort(corrected, axis=1)
    entropy = -(corrected * np.log(np.clip(corrected, 1e-12, 1))).sum(1)
    weight = ordered[:, -1] * (ordered[:, -1] - ordered[:, -2]) * (1 - entropy / np.log(corrected.shape[1]))
    accepted = corrected.max(1) >= 0.7
    pseudo = corrected.argmax(1)
    result = []
    for c in range(corrected.shape[1]):
        w = weight * accepted * (pseudo == c)
        result.append(fallback[c] if w.sum() <= 1e-8 else normalize(((w[:, None] * x).sum(0) / w.sum())[None])[0])
    return normalize(np.stack(result))


def predict(method, sx, sy, tx, text, per_prompt):
    source = source_proto(sx, sy, text)
    if method == "clip_zero_shot": prototype = normalize(per_prompt[:, 0])
    elif method == "prompt_ensemble": prototype = text
    else:
        source_weight = 0.0 if method == "no_source_anchor" else 0.1
        base = normalize((1 - source_weight) * text + source_weight * source)
        target = target_proto(tx, classify(tx, base), base)
        prototype = normalize((1 - source_weight - 0.025) * text + source_weight * source + 0.025 * target)
    return classify(tx, prototype)


def load(name):
    with np.load(WORK / f"{name}_vitb16_openai.npz", allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def labels(name):
    with np.load(WORK / f"{name}_vitb16_openai.source_labels.npz", allow_pickle=False) as d:
        return d["labels"].astype(np.int64)


def evaluate_benchmark(name, tasks):
    records = []
    for task, source_name, target_name in tasks:
        source, target = load(source_name), load(target_name)
        source_labels = labels(source_name)
        for method in ("clip_zero_shot", "prompt_ensemble", "no_source_anchor", "satpa"):
            probabilities = predict(method, source["image_features"], source_labels, target["image_features"],
                                    target["text_features"], target["text_features_per_prompt"])
            prediction_path = WORK / f"{name}_{task}_{method}_predictions.npz"
            np.savez_compressed(prediction_path, sample_ids=target["sample_ids"], probabilities=probabilities,
                                predictions=probabilities.argmax(1))
            target_labels = labels(target_name)  # loaded only after predictions are saved
            records.append({"benchmark": name, "task": task, "method": method,
                            "accuracy": accuracy_score(target_labels, probabilities.argmax(1)) * 100,
                            "macro_f1": f1_score(target_labels, probabilities.argmax(1), average="macro") * 100})
    frame = pd.DataFrame(records)
    frame.to_csv(WORK / f"{name}_vitb16_results.csv", index=False)
    return frame


gpu_count = max(1, torch.cuda.device_count())
JOBS = resolve_jobs()
if gpu_count >= 2:
    assignments = [JOBS[:3] + [JOBS[5]], [JOBS[3], JOBS[4], JOBS[6]]]
    mp.start_processes(extract_worker, args=(2, assignments), nprocs=2, join=True, start_method="fork")
else:
    extract_worker(0, 1, [JOBS])

office31_tasks = [("A2W","office31_amazon","office31_webcam"),("D2W","office31_dslr","office31_webcam"),
                  ("W2A","office31_webcam","office31_amazon"),("A2D","office31_amazon","office31_dslr"),
                  ("D2A","office31_dslr","office31_amazon"),("W2D","office31_webcam","office31_dslr")]
officehome_tasks = [(a+b, "officehome_"+s, "officehome_"+t) for a,s in (("A2","art"),("C2","clipart"),("P2","product"),("R2","real_world"))
                    for b,t in (("A","art"),("C","clipart"),("P","product"),("R","real_world")) if s != t]
o31 = evaluate_benchmark("office31", office31_tasks)
oh = evaluate_benchmark("officehome", officehome_tasks)
summary = pd.concat([o31, oh]).groupby(["benchmark","method"])[["accuracy","macro_f1"]].mean().reset_index()
summary.to_csv(WORK / "vitb16_summary.csv", index=False)
print(summary.to_string(index=False))
subprocess.check_call(["zip", "-qr", "/kaggle/working/satpa_vitb16_outputs.zip", str(WORK)])
