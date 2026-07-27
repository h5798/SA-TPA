from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((PROJECT / "protocols" / "locked_hyperparameters.json").read_text(encoding="utf-8"))
    locked_bytes = subprocess.check_output(
        ["git", "-C", str(PROJECT), "show", f"{lock['git_commit']}:configs/main.yaml"]
    )
    locked_hash = hashlib.sha256(locked_bytes).hexdigest()
    if locked_hash != lock["config_sha256"]:
        raise RuntimeError("The configuration stored at the locked commit does not match its hash")
    locked_config = yaml.safe_load(locked_bytes)
    current_config = yaml.safe_load((PROJECT / "configs" / "main.yaml").read_text(encoding="utf-8"))
    for section in ("model", "method"):
        if locked_config[section] != current_config[section]:
            raise RuntimeError(f"Locked experimental section changed: {section}")
    print(json.dumps({
        "lock_valid": True,
        "locked_commit": lock["git_commit"],
        "locked_config_sha256": locked_hash,
        "current_model_and_method_semantically_identical": True,
    }, indent=2))


if __name__ == "__main__":
    main()

