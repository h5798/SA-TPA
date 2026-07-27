from __future__ import annotations

import csv
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT = Path("D:/456/project")
RESULTS = Path("D:/456/results")
TAG = "experiments-final-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(PROJECT), *args], text=True).strip()


def main():
    tagged_commit = git("rev-list", "-n", "1", TAG)
    csv_files = sorted(RESULTS.rglob("*.csv"))
    rows = [{
        "relative_path": path.relative_to(Path("D:/456")).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    } for path in csv_files]
    manifest_csv = RESULTS / "audits" / "result_csv_sha256_manifest.csv"
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Experiment freeze manifest",
        "",
        f"- Freeze tag: `{TAG}`",
        f"- Tagged experiment commit: `{tagged_commit}`",
        f"- Manifest generated: `{datetime.now().astimezone().isoformat()}`",
        "- Final method: original fixed-weight SA-TPA",
        "- Fusion: text/source/target = 0.875/0.100/0.025",
        "- Backbone evidence: OpenAI CLIP ViT-B/32 and ViT-B/16",
        "",
        "## Rejected exploratory extensions",
        "",
        "- Class-adaptive fusion: rejected.",
        "- Cross-prototype agreement filtering: rejected after cross-dataset regression.",
        "- Iterative target update: rejected after no gain.",
        "- SPT-SA optimal transport: rejected at the preregistered development gate.",
        "",
        "## Result CSV integrity",
        "",
        f"The SHA256 manifest contains {len(rows)} CSV files and is stored at "
        "`D:/456/results/audits/result_csv_sha256_manifest.csv`.",
        "",
        "## Audit references",
        "",
        "- `D:/456/project/docs/targeted_improvement_audit.md`",
        "- `D:/456/project/docs/spt_sa_experiment_audit.md`",
        "- `D:/456/project/docs/baseline_compatibility.md`",
        "- `D:/456/project/docs/data_replica_audit.md`",
    ]
    (PROJECT / "docs" / "experiment_freeze_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tagged commit: {tagged_commit}; hashed CSV files: {len(rows)}")


if __name__ == "__main__":
    main()
