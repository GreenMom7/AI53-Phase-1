# AI53 - MLOps Project Phase 1 Lab Report

## Milestone 0 - Starter Audit

### Starter Issues Table

| #   | File / Setting                            | Problem Description                                                                                                                                                                                                                                                                                                                                                                                                                              | Suggested Fix                                                                                                                                                                                                                    |
| --- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `scripts/check_environment.py`            | `Path(sys.executable).resolve().is_relative_to((root / '.venv').resolve())` follows symlinks. On macOS/Linux, `uv` creates `.venv/bin/python` as a symlink to the base Python install, causing the environment check to falsely fail with "Interpreter is outside the project .venv".                                                                                                                                                            | Change the check to inspect `sys.prefix` directly (`Path(sys.prefix).resolve() == (root / '.venv').resolve()`) or use `Path(sys.executable).is_relative_to(root / '.venv')` without dereferencing symlinks.                      |
| 2   | `.gitignore`                              | Missing or incomplete `.gitignore`. Large dataset archives (`dataset-archives/*.zip`, ~180 MB total, with `train.zip` alone being 143.8 MB) and raw image directories (`data/raw/`) are staged by default when running `git add .`, which exceeds GitHub's 100 MB per-file upload limit and bloats the Git repository.                                                                                                                           | Create a `.gitignore` file that ignores `dataset-archives/`, `data/raw/`, `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `mlflow.db`, and `mlruns/`.                                                                       |
| 3   | `src/reproducibility.py`                  | Incomplete RNG seeding in starter: only Python's standard `random.seed(seed)` was called. NumPy (`np.random.seed`), PyTorch CPU (`torch.manual_seed`), PyTorch CUDA (`torch.cuda.manual_seed_all`), and cuDNN determinism flags (`deterministic = True`, `benchmark = False`) were unconfigured, leading to non-repeatable weights and data shuffling. DataLoader worker seeding and deterministic generator generation were also unimplemented. | Fully implement `set_reproducibility(seed)` to seed Python, NumPy, PyTorch CPU/CUDA, and configure cuDNN flags. Implement `seed_worker` using `torch.initial_seed()` and `make_generator(seed)` using `torch.Generator`.         |
| 4   | `src/validate_data.py`                    | No automated data quality gate implemented before training. Corrupted images, missing class subfolders, split mismatch, or empty classes would go undetected until training crashes or produces silent label drift.                                                                                                                                                                                                                              | Implement `validate_dataset()` to verify presence of train/validation/test splits, verify class labels, validate expected sample counts against course manifest, and verify that Pillow decodes all image pixels without errors. |
| 5   | `params.yaml` / Test Set Evaluation       | `selection.run_id` was unpopulated, and evaluation lacked strict enforcement against using the test set during iterative tuning. Over-evaluating or tuning hyperparameters against the test set causes data leakage and optimistic performance bias.                                                                                                                                                                                             | Restrict iterative model training and hyperparameter comparisons exclusively to the validation split. Keep the test set isolated until final handoff evaluation on the single selected model (`selection.run_id`).               |
| 6   | `src/provenance.py` / Model Checkpointing | Starter code did not record end-to-end lineage (exact Git commit SHA, DVC data hash, hyperparameters, training environment) linked to `model.pt`.                                                                                                                                                                                                                                                                                                | Record Git commit hash, dataset hash/version, and MLflow run metadata alongside the saved model checkpoint so every artifact has full upstream traceability.                                                                     |

---

### Engineering Checkpoint 1

**Question:** _Why is a working `model.pt` insufficient evidence that an experiment is reproducible?_

**Answer:**
A trained model weight file (`model.pt`) is merely a static binary artifact representing the final numerical state of the neural network weights. It does not provide any verifiable evidence of reproducibility because:

1. **Missing Code Lineage:** It does not record the exact version of the training code or repository state (e.g., Git commit hash, uncommitted local modifications).
2. **Missing Data Provenance:** It does not record the exact dataset snapshot, preprocessing transformations, data splits, or whether data leakage occurred.
3. **Missing Environment Specifications:** It contains no record of the runtime dependencies (Python version, library versions in `uv.lock`, CUDA/cuDNN versions, or OS platform). Minor version discrepancies in deep learning frameworks can alter training dynamics.
4. **Missing Stochastic Controls:** It provides no guarantee that random seeds (Python, NumPy, PyTorch, cuDNN, DataLoader worker shuffling) were properly pinned or that the experiment can produce identical results when re-run.
5. **No Execution Traceability:** Without logged parameters, metrics history, and runtime logs (e.g., in MLflow and DVC), an external reviewer cannot audit whether the artifact was generated legitimately or manually swapped.

---

## Milestone 1 - Repeatable Training

### Roles of Environment Components

- **`pyproject.toml`:** Declares the high-level project metadata, build system, and abstract package dependencies (e.g., `torch`, `torchvision`, `mlflow`, `dvc`, `pytest`). It allows us to know what libraries it needs.
- **`uv.lock`:** Records the exact, cryptographically hashed dependency resolution tree (exact versions, specific wheels, platform tags, transitive dependencies). It guarantees deterministic, cross-platform resolution so that every installation uses identical packages.
- **`.venv/`:** The local virtual environment directory containing the isolated Python interpreter and installed binary packages. It isolates the project's runtime environment from the system Python and is never committed to Git.

### Engineering Checkpoint 2

**Question:** _Does random_state=42 guarantee bit-for-bit identical results on every CPU/GPU/soft-
ware stack? Explain briefly._

**Answer:**

## Milestone 3 - Extract and track the images

### Engineering Checkpoint 3

## Milestone 4 -

### Engineering Checkpoint

## Milestone 5 - ...

### Engineering Checkpoint

## Milestone 6 - ...

### Engineering Checkpoint

## Milestone 7 - ...

### Engineering Checkpoint
