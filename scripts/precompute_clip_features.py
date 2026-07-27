from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROMPTS = (
    "a photo of a {}.",
    "a product photo of a {}.",
    "an image of a {}.",
    "a close-up photo of a {}.",
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


class UnlabeledImageDataset(Dataset):
    """Load images without returning or encoding instance labels."""

    def __init__(self, root: str, transform):
        self.root = Path(root)
        self.paths = sorted(
            path for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No images found below {self.root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def readable(name: str) -> str:
    aliases = {"back_pack": "backpack", "bike": "bicycle", "phone": "telephone"}
    return aliases.get(name, name.replace("_", " "))


def main(args):
    cache = Path(args.cache_root)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["HF_HOME"] = str(cache / "huggingface")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.weights, cache_dir=str(cache / "open_clip")
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    class_root = Path(args.class_root or args.domain_root)
    classes = sorted(path.name for path in class_root.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No class directories found below {class_root}")

    text_features = []
    text_features_per_prompt = []
    with torch.no_grad():
        for class_name in classes:
            texts = [prompt.format(readable(class_name)) for prompt in PROMPTS]
            features = model.encode_text(tokenizer(texts).to(device))
            features = features / features.norm(dim=1, keepdim=True)
            text_features_per_prompt.append(features)
            prototype = features.mean(dim=0)
            text_features.append(prototype / prototype.norm())
        text_features = torch.stack(text_features)
        text_features_per_prompt = torch.stack(text_features_per_prompt)

    dataset = UnlabeledImageDataset(args.domain_root, preprocess)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    images, probabilities = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=Path(args.domain_root).name):
            features = model.encode_image(batch.to(device, non_blocking=True))
            features = features / features.norm(dim=1, keepdim=True)
            probs = (100.0 * features @ text_features.T).softmax(dim=1)
            images.append(features.cpu().numpy().astype(np.float16))
            probabilities.append(probs.cpu().numpy().astype(np.float16))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_features = np.concatenate(images)
    probs = np.concatenate(probabilities)
    np.savez_compressed(
        output,
        image_features=image_features,
        text_features=text_features.cpu().numpy().astype(np.float16),
        text_features_per_prompt=text_features_per_prompt.cpu().numpy().astype(np.float16),
        probabilities=probs,
        predictions=probs.argmax(1),
        confidences=probs.max(1),
        classes=np.asarray(classes),
        sample_ids=np.arange(len(dataset), dtype=np.int32),
    )
    if args.write_source_labels:
        class_to_index = {name: index for index, name in enumerate(classes)}
        source_labels = np.asarray(
            [class_to_index[path.parent.name] for path in dataset.paths], dtype=np.int16
        )
        sidecar = output.with_suffix(".source_labels.npz")
        np.savez_compressed(sidecar, sample_ids=np.arange(len(dataset)), labels=source_labels)
    metadata = {
        "model": args.model,
        "weights": args.weights,
        "open_clip": open_clip.__version__,
        "samples": len(dataset),
        "classes": len(classes),
        "domain_root": str(Path(args.domain_root).resolve()),
        "instance_paths_encoded_in_feature_file": False,
        "instance_labels_encoded_in_feature_file": False,
        "source_label_sidecar_written": bool(args.write_source_labels),
        "output": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-root", required=True)
    parser.add_argument(
        "--class-root",
        help="Domain whose directory names define the known shared class vocabulary.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-root", default="D:/456/cache")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--weights", default="openai")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--write-source-labels",
        action="store_true",
        help="Write labels to a separate source-only sidecar; never load it for a target domain.",
    )
    main(parser.parse_args())
