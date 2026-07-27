from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("D:/456")


def means(path):
    frame = pd.read_csv(path)
    return frame.groupby("method")["accuracy"].mean().to_dict()


def main():
    office31 = means(ROOT / "results/office31/development_v2.csv")
    officehome = means(ROOT / "results/officehome/confirmatory_v1.csv")
    office31_extra = means(ROOT / "results/office31/additional_baselines_v1.csv")
    officehome_extra = means(ROOT / "results/officehome/additional_baselines_v1.csv")
    office31_vitb16 = means(ROOT / "results/vitb16/office31_vitb16_results.csv")
    officehome_vitb16 = means(ROOT / "results/vitb16/officehome_vitb16_results.csv")
    home = pd.read_csv(ROOT / "results/officehome/confirmatory_v1.csv").pivot(
        index="task", columns="method", values="accuracy"
    )
    delta_zero = home.satpa - home.clip_zero_shot
    delta_prompt = home.satpa - home.prompt_ensemble
    delta_anchor = home.satpa - home.no_source_anchor
    summary = {
        "topic": "Backpropagation-Free Source-Anchored Tri-Prototype Adaptation for Vision-Language Models",
        "acronym": "SA-TPA",
        "backbone": "OpenAI CLIP ViT-B/32",
        "office31_mean_accuracy": {**office31, **office31_extra},
        "officehome_mean_accuracy": {**officehome, **officehome_extra},
        "vitb16_office31_mean_accuracy": office31_vitb16,
        "vitb16_officehome_mean_accuracy": officehome_vitb16,
        "officehome_gates": {
            "mean_delta_vs_zero_shot": float(delta_zero.mean()),
            "mean_delta_vs_prompt_ensemble": float(delta_prompt.mean()),
            "mean_delta_vs_no_source_anchor": float(delta_anchor.mean()),
            "tasks_declining_more_than_0_5_vs_zero_shot": int((delta_zero < -0.5).sum()),
            "worst_delta_vs_zero_shot": float(delta_zero.min()),
            "full_outperforms_no_source_anchor_on_average": bool(delta_anchor.mean() > 0),
            "predeclared_gate_passed": bool(
                delta_zero.mean() >= 1.0
                and (delta_zero < -0.5).sum() <= 4
                and delta_zero.min() >= -1.5
                and delta_anchor.mean() > 0
            ),
        },
        "locked_protocol": json.loads(
            (ROOT / "project/protocols/locked_hyperparameters.json").read_text(encoding="utf-8")
        ),
    }
    output = ROOT / "results/experiment_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
