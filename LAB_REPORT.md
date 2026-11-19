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

**Question:** Why is a working `model.pt` insufficient evidence that an experiment is reproducible?

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

**Question:** Does random_state=42 guarantee bit-for-bit identical results on every CPU/GPU/soft-
ware stack? Explain briefly.

**Answer:** [ANSWER NEEDED]

## Milestone 3 - Extract and track the images

### Engineering Checkpoint 3

**Question:** _In one sentence each, state what the Git remote stores, what the DVC remote stores,
and whether git push transfers the DVC-managed raw images._

**Answer:**

- **Git remote (GitHub):** stores the code, configuration, commit history and the small `.dvc` pointer files, not the images themselves.
- **DVC remote:** stores the actual raw image files (the large data), identified by the hash in `data/raw.dvc`.
- **Does `git push` transfer the raw images?** No. It only sends the Git history, including the pointer file. The images are sent separately with `dvc push`.

## Milestone 4 - DVC Pipeline Design

### Questions

- Should PREPARE depend on `reports/data_quality.json`? : Yes, in case of any presence of corrupt/bad data, the pipeline stops allwoing us to fix the issue before going to the training process.
- Should PREPARE also depend directly on `data/raw`? : Yes, because the `prepare` process reads the images.
- Which `params.yaml` changes should make TRAIN rerun? : It will rerun everytime any key under `params:` changes
- Why should EVALUATE track `selection.run_id`? : Because the ID belongs to specific one model.
- Which file holds the metrics? : Metric file will be `reports/test_metrics.json`

### Stages

| Stage    | cmd                                           | deps                                                                                                                                                                                                                                                            | params                                                                                                                                                                                                                                                                | outs                                                                                                        | metrics                     |
| -------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------- |
| validate | `uv run --locked python -m src.validate_data` | `data/raw`, `src/validate_data.py`, `uv.lock`, `pyproject.toml`                                                                                                                                                                                                 | -                                                                                                                                                                                                                                                                     | `reports/data_quality.json`                                                                                 | -                           |
| prepare  | `uv run --locked python -m src.prepare`       | `reports/data_quality.json`, `data/raw`, `src/prepare.py`, `uv.lock`, `pyproject.toml`                                                                                                                                                                          | `data.image_size`                                                                                                                                                                                                                                                     | `data/processed`, `reports/data_summary.json`                                                               | -                           |
| train    | `uv run --locked python -m src.train`         | `data/processed/train`, `data/processed/validation`, `data/processed/manifests/train.csv`, `data/processed/manifests/validation.csv`, `src/train.py`, `src/data.py`, `src/model.py`, `src/provenance.py`, `src/reproducibility.py`, `uv.lock`, `pyproject.toml` | `train.random_state`, `train.batch_size`, `train.dropout`, `train.learning_rate`, `train.epochs`, `data.image_size`, `compute.device`, `augmentation.enabled`, `augmentation.horizontal_flip`, `augmentation.rotation_degrees`, `tracking.uri`, `tracking.experiment` | `models/model.pt`, `reports/train_summary.json`, `reports/training_curves.png`, `reports/run_manifest.json` | -                           |
| evaluate | `uv run --locked python -m src.evaluate`      | `data/processed/test`, `data/processed/manifests/train.csv`, `data/processed/manifests/validation.csv`, `data/processed/manifests/test.csv`, `src/evaluate.py`, `src/data.py`, `src/provenance.py`, `src/reproducibility.py`, `uv.lock`, `pyproject.toml`       | `selection.run_id`, `data.image_size`, `compute.device`, `evaluation.batch_size`, `tracking.uri`, `tracking.experiment`                                                                                                                                               | `reports/test_confusion_matrix.png`                                                                         | `reports/test_metrics.json` |

- Output of `dvc stage list`:
  ```validate Outputs reports/data_quality.json
  prepare   Outputs data/processed, reports/data_summary.json
  train     Outputs models/model.pt, reports/train_summary.json, reports/training_curves.pn…
  evaluate  Outputs reports/test_confusion_matrix.png; Reports reports/test_metrics.json
  ```
- Output of `dvc dag`:

```
        +--------------+
        | data/raw.dvc |
        +--------------+
           **        **
         **            **
        *                **
+----------+               *
| validate |             **
+----------+           **
           **        **
             **    **
               *  *
           +---------+
           | prepare |
           +---------+
           **        **
         **            **
        *                *
  +-------+         +----------+
  | train |         | evaluate |
  +-------+         +----------+
```

- Does EVALUATE avoid depending on TRAIN? : Yes, `evaluate` branches directly from `prepare`, completely decoupled from the `train` stage.

## Milestone 5 - MLflow Run Records

Result of the full `uv run --locked pytest -q` run:

```
19 passed in 3.68s
```

### Engineering Checkpoint 4

**Question:** In one sentence per tool, explain what Git, DVC and MLflow each record about a training Run.

**Answer:**

- Git: records the exact code, configuration (`params.yaml`, `dvc.yaml`) and the small `.dvc` pointer files at a given commit, so the Run's source can be restored by its commit SHA
- DVC: records the data versions and pipeline state (hashes of inputs, parameters and outputs in `dvc.lock`), so it knows which data produced which outputs and which stages need to rerun
- MLflow: records the Run itself (its parameters, per-epoch metrics, training time, artifacts (reports, run manifest), the logged model with its signature, and the dataset input)

## Milestone 6 - Which Stages Rerun

### Decisions made before training

- Main validation metric: -
- Tie-break rule: -
- What I expect a lower learning rate to change: -

### Prediction table (before running)

| Change                       | VALIDATE | PREPARE | TRAIN  | EVALUATE     |
| ---------------------------- | -------- | ------- | ------ | ------------ |
| Change `train.learning_rate` | REUSED   | REUSED  | RERUN  | REUSED       |
| Change `data.image_size`     | REUSED   | RERUN   | RERUN  | NEEDS UPDATE |
| Edit `src/validate_data.py`  | RERUN    | REUSED  | REUSED | REUSED       |
| Edit only `README.md`        | REUSED   | REUSED  | REUSED | REUSED       |

### Run A (baseline)

- MLflow Run ID: 6fa3cb3ceba44c0ba38a5802b7ef9502

### Actual results table (after running)

| Check                                  | VALIDATE | PREPARE | TRAIN  | EVALUATE | Terminal lines                      |
| -------------------------------------- | -------- | ------- | ------ | -------- | ----------------------------------- |
| 1. Rerun with no change                | REUSED   | REUSED  | REUSED | NOT RUN  | `Data and pipelines are up to date` |
| 2. Comment added to `validate_data.py` | RERUN    | REUSED  | REUSED | NOT RUN  | `Running stage 'validate'`          |
| 3. Learning rate set to 0.0003 (Run B) | REUSED   | REUSED  | RERUN  | NOT RUN  | `Running stage 'train'`             |

- Run B MLflow Run ID: 0d5f0c5d68674d5d803d1016c087b26f
- Differences between predictions and results: The actual behavior matches the predictions in the first table

### Engineering Checkpoint 5

**Explanation of at least two rows (one setting change, one file edit):**

- Setting change row: input that changes per stage:
  `validate` and `prepare` do not track `train.learning_rate` in `dvc.yaml` under `params` so their hashes remain unchanged. `train` explicitly declares `train.learning_rate` under `params` which triggers re-execution
- File edit row: why the inputs stay the same (or change) per stage: `validate` lists `src/validate_data.py` under `deps` so modifying the file forces validate to rerun. However, because the comment does not change the generated `reports/data_quality.json`, its content hash remains identical.

---

## Milestone 7 - Controlled Experiments

### Run C

- Single change chosen (dropout / augmentation / batch size): -
- What I expect and why: -
- MLflow Run ID: -

### Comparison table (last-epoch validation scores)

| Run | Controlled change | Val. Accuracy | Val. Macro-F1 | Training time | MLflow Run ID |
| --- | ----------------- | ------------- | ------------- | ------------- | ------------- |
| A   | Baseline          | -             | -             | -             | -             |
| B   | -                 | -             | -             | -             | -             |
| C   | -                 | -             | -             | -             | -             |

- B vs A: does the result support my prediction?: -
- C vs A: does the result support my prediction?: -
- Limitation (one seed, small validation set): -

### Engineering Checkpoint 6

- Why my primary metric suits this three-class problem: -
- Why the test set is used only after choosing a model: -

---

## Milestone 8 - Select and Test One Model

### Selection

- Selected Run ID: -
- Score on the main metric: -
- Other evidence used: -
- One trade-off or limitation: -

### Final test results

- Run ID in `reports/test_metrics.json`: -
- Test accuracy / macro-F1: -
- Notes on the confusion matrix: -

### Final checks

- `pytest -q` result: -
- `dvc status` result: -
- GitHub repository URL: -
