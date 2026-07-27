"""Create a SHA256 inventory for the final non-writing deliverables."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PROJECT = Path(r"D:\456\project")
OUTPUT = PROJECT / "outputs" / "final_assets" / "deliverables_sha256_manifest.csv"

INCLUDED_DOCS = [
    PROJECT / "docs" / "experiment_freeze_manifest.md",
    PROJECT / "docs" / "bootstrap_audit.md",
    PROJECT / "docs" / "data_replica_audit.md",
    PROJECT / "docs" / "claims_evidence_index.md",
    PROJECT / "docs" / "final_non_writing_completion_report.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect() -> list[Path]:
    files: list[Path] = []
    files.extend(path for path in INCLUDED_DOCS if path.is_file())
    for directory in (PROJECT / "tables", PROJECT / "figures"):
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    workbook = PROJECT / "outputs" / "final_assets" / "SA-TPA_Final_Experiment_Assets.xlsx"
    if workbook.is_file():
        files.append(workbook)
    return sorted(set(files), key=lambda path: path.as_posix().lower())


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in collect():
        rows.append(
            {
                "relative_path": path.relative_to(PROJECT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} hashes to {OUTPUT}")


if __name__ == "__main__":
    main()

