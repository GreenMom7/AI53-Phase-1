from __future__ import annotations

import hashlib
import json
import importlib.metadata
import platform
import subprocess
import torch
from pathlib import Path

# helper functions for git and package version
def get_git_info() -> tuple[str, bool]:
    """Safely retrieve Git commit SHA and working-tree dirty status."""
    try:
        proc_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commit = proc_commit.stdout.strip() if proc_commit.returncode == 0 else "unavailable"

        proc_status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        dirty = bool(proc_status.stdout.strip()) if proc_status.returncode == 0 else False
    except Exception:
        commit = "unavailable"
        dirty = False

    return commit, dirty


def get_version(package_name: str) -> str:
    """Safely query package version via importlib.metadata."""
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return "unavailable"

def training_data_identity(
    train_manifest: Path = Path("data/processed/manifests/train.csv"),
    validation_manifest: Path = Path("data/processed/manifests/validation.csv"),
) -> str:
    """Return a deterministic identity for the training + validation manifests.

    This helper is provided because the final evaluator uses the same identity
    to verify that the selected MLflow Run matches the current prepared data.
    Do not hard-code the value.
    """
    paths = (train_manifest, validation_manifest)
    if any(not path.exists() for path in paths):
        return "unavailable"

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Build experiment provenance metadata.

    TODO: collect at least:
      - Git commit SHA and whether the working tree is dirty;
      - identity of the prepared training + validation manifests;
      - Python, PyTorch, torchvision, DVC and MLflow versions;
      - OS/platform and CUDA runtime (when available);
      - compute device; random seed; image size; training parameters;
      - MLflow Run ID.

    Hints: subprocess, platform, importlib.metadata, torch,
    training_data_identity().
    """
    commit, dirty = get_git_info()
    return {
        "git": {
            "commit": commit,
            "dirty": dirty,
        },
        "data": {
            "training_data_identity": training_data_identity(),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": get_version("torch"),
            "torchvision": get_version("torchvision"),
            "dvc": get_version("dvc"),
            "mlflow": get_version("mlflow"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "configuration": {
            "image_size": int(params.get("data", {}).get("image_size", 96)),
            "batch_size": int(params.get("train", {}).get("batch_size", 32)),
            "epochs": int(params.get("train", {}).get("epochs", 4)),
            "learning_rate": float(params.get("train", {}).get("learning_rate", 0.001)),
            "dropout": float(params.get("train", {}).get("dropout", 0.3)),
            "augmentation": params.get("augmentation", {}),
        },
        "execution": {
            "device": str(device),
            "random_state": int(params.get("train", {}).get("random_state", 42)),
            "mlflow_run_id": str(mlflow_run_id),
        },
    }


def write_run_manifest(params: dict, device: str, mlflow_run_id: str, path: Path) -> dict:
    manifest = build_run_manifest(params, device, mlflow_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
