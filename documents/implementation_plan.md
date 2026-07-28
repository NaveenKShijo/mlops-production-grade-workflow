# CI/CD Pipeline, Dockerfiles, MLflow Registry & Tests — Full Implementation Plan

## Background

You have an ML project that trains a `LinearRegression` model on `insurance.csv` (versioned with DVC on S3) and serves predictions via FastAPI on SageMaker. The current codebase has placeholder Dockerfiles, a partially complete `cd.yaml` with syntax errors, and empty test directories. This plan addresses **everything** end-to-end.

---

## Corrections to Your Current Understanding

> [!IMPORTANT]
> **Training image — You are correct.** The training Docker image should be a **base image** (runtime only: Python, git, dvc, AWS CLI, pip packages). The repo + dataset are fetched dynamically at container startup via an entrypoint script. This keeps the image **reusable across commits** without rebuilding for every code change.

> [!IMPORTANT]
> **Inference image — You were partially wrong, and here's why:**
>
> Unlike training, the inference image in production **should NOT clone the repo at startup**. Best practice is:
> 1. The inference **source code** (`main.py`, `predict.py`) is **baked into the image** at build time (via `COPY`), because the serving code IS the product.
> 2. The **model artifacts** (`model.pkl`, `scaler.pkl`) are **NOT baked in**. SageMaker automatically extracts `model.tar.gz` from S3 and mounts it at `/opt/ml/model/` before your container starts.
>
> **Why?** If you clone at startup:
> - You introduce a git/network dependency on every cold start (fragile)
> - You can't guarantee which commit gets deployed (race condition)
> - Rollbacks become impossible — you need to redeploy with a specific git sha
>
> The correct pattern: build a **tagged, immutable** inference image per commit. SageMaker handles model delivery. The image tag acts as your version contract.

> [!WARNING]
> **File organization issue:** The existing `Dockerfile.training` and `Dockerfile.inference` are at the project root, which is messy. Move them under a `docker/` directory for clarity. The existing `implementation_plan.md` already proposed this — this plan follows through.

---

## Current Issues Found

| File | Issue |
|------|-------|
| [cd.yaml](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/.github/workflows/cd.yaml) | Uses `runs:` instead of `run:` (YAML syntax error). Missing env vars for deploy step. Missing `-f Dockerfile.inference`. Uses `${{DOCKER_USERNAME}}` instead of `${{secrets.DOCKER_USERNAME}}`. |
| [ci.yaml](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/.github/workflows/ci.yaml) | `python-version: python:3.11` should be `"3.11"`. No `-f Dockerfile.training` flag. Missing DVC credentials for training container. |
| [deploy.py](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/src/pipeline/deploy.py) | Only has comments, no implementation. |
| [main.py](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/src/inference/main.py) | Uses `pd.DataFrame` but imports `pandas` (not as `pd`). Missing return statement in `/invocations`. |
| [train.py](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/src/training/train.py) | Uses `TRAINING_JOB_RUN` env var but `run_training.py` sets `TRAINING_JOB_NAME` — mismatch. |
| Root Dockerfiles | `Dockerfile.training` and `Dockerfile.inference` are empty placeholders at root. |
| Tests | `tests/training/` and `tests/inference/` are empty directories. |

---

## Proposed Changes

### File Organization (After Changes)

```
LProj1/
├── .github/workflows/
│   ├── ci.yaml                          ← REWRITE
│   └── cd.yaml                          ← REWRITE
├── docker/
│   ├── training/
│   │   ├── Dockerfile                   ← NEW (base image)
│   │   └── entrypoint.sh               ← NEW (git clone + dvc pull + train)
│   └── inference/
│       ├── Dockerfile                   ← NEW (baked code, no model)
│       └── entrypoint.sh               ← NEW (starts uvicorn)
├── src/
│   ├── training/
│   │   └── train.py                     ← FIX env var mismatch
│   ├── inference/
│   │   ├── main.py                      ← FIX import + return
│   │   └── predict.py                   ← NO CHANGE ✅
│   └── pipeline/
│       ├── run_training.py              ← MINOR UPDATE (pass DVC creds)
│       └── deploy.py                    ← REWRITE (full implementation)
├── tests/
│   ├── training/
│   │   ├── __init__.py                  ← NEW
│   │   └── test_train.py               ← NEW
│   ├── inference/
│   │   ├── __init__.py                  ← NEW
│   │   └── test_inference.py            ← NEW
│   └── conftest.py                      ← NEW (shared fixtures)
├── requirements.txt                     ← UPDATE (add test deps)
├── requirements_train.txt               ← NEW (container pip deps)
├── requirements_inference.txt           ← NEW (container pip deps)
├── Dockerfile.training                  ← DELETE (moved to docker/)
├── Dockerfile.inference                 ← DELETE (moved to docker/)
├── data/
│   └── insurance.csv.dvc
└── .gitignore                           ← UPDATE
```

---

### Docker — Training

#### [NEW] [Dockerfile](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/docker/training/Dockerfile)

**Base image only** — no repo code, no data. Contains Python runtime, git, DVC, AWS CLI, and pip packages. The entrypoint clones the repo and pulls data at runtime.

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

COPY requirements_train.txt /tmp/requirements_train.txt
RUN pip install --no-cache-dir -r /tmp/requirements_train.txt

COPY docker/training/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/entrypoint.sh"]
```

#### [NEW] [entrypoint.sh](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/docker/training/entrypoint.sh)

Runs inside SageMaker. Clones repo at the specific commit, pulls dataset via DVC, then trains.

```bash
#!/bin/bash
set -e

echo "=== Cloning repository at commit ${GIT_COMMIT_SHA:-HEAD} ==="
git clone https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git /opt/ml/code
cd /opt/ml/code
if [ -n "$GIT_COMMIT_SHA" ]; then
    git checkout "$GIT_COMMIT_SHA"
fi

echo "=== Pulling dataset via DVC ==="
dvc pull data/insurance.csv

echo "=== Starting training ==="
python src/training/train.py
```

#### [NEW] [requirements_train.txt](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/requirements_train.txt)

```
numpy
pandas
matplotlib
joblib
scikit-learn
mlflow
dvc[s3]
boto3
```

---

### Docker — Inference

#### [NEW] [Dockerfile](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/docker/inference/Dockerfile)

**Bakes the inference source code into the image.** Model artifacts (`model.pkl`, `scaler.pkl`) are NOT included — SageMaker extracts `model.tar.gz` to `/opt/ml/model/` at deploy time.

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

COPY requirements_inference.txt /tmp/requirements_inference.txt
RUN pip install --no-cache-dir -r /tmp/requirements_inference.txt

# Bake inference code into image (this IS the product)
COPY src/inference/ /opt/ml/code/

COPY docker/inference/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/entrypoint.sh"]
```

#### [NEW] [entrypoint.sh](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/docker/inference/entrypoint.sh)

SageMaker has already extracted `model.tar.gz` → `/opt/ml/model/` before this runs.

```bash
#!/bin/bash
set -e

echo "=== Model artifacts at /opt/ml/model/ ==="
ls -la /opt/ml/model/

echo "=== Starting FastAPI server on port 8080 ==="
cd /opt/ml/code
uvicorn main:app --host 0.0.0.0 --port 8080
```

#### [NEW] [requirements_inference.txt](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/requirements_inference.txt)

```
numpy
pandas
joblib
scikit-learn
fastapi
uvicorn
```

---

### GitHub Actions — CI

#### [MODIFY] [ci.yaml](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/.github/workflows/ci.yaml)

Key changes:
- Fix `python-version` format
- Use `-f docker/training/Dockerfile` 
- Pass `GITHUB_TOKEN`, `GITHUB_REPO`, `GIT_COMMIT_SHA` and DVC S3 credentials to the training container so it can clone the repo and `dvc pull`
- DockerHub used (per your existing setup — switch to ECR when ready)

---

### GitHub Actions — CD

#### [MODIFY] [cd.yaml](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/.github/workflows/cd.yaml)

Key changes:
- Fix all syntax errors (`runs:` → `run:`, missing `secrets.` prefix)
- Use `-f docker/inference/Dockerfile`
- Add all env vars for deploy step
- Add inference tests before deployment

---

### Pipeline Scripts

#### [MODIFY] [deploy.py](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/src/pipeline/deploy.py)

Full implementation: queries MLflow Registry for "Production" model → derives `model.tar.gz` S3 URI → deploys/updates SageMaker Endpoint.

#### [MODIFY] [run_training.py](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/src/pipeline/run_training.py)

Pass `GITHUB_TOKEN`, `GITHUB_REPO`, `GIT_COMMIT_SHA` as environment variables to the SageMaker container so the entrypoint can clone the correct commit. Pass DVC S3 credentials.

---

### Source Code Fixes

#### [MODIFY] [train.py](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/src/training/train.py)

Fix: `TRAINING_JOB_RUN` → `TRAINING_JOB_NAME` (match what `run_training.py` passes).

#### [MODIFY] [main.py](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/src/inference/main.py)

Fix: `import pandas` → `import pandas as pd`. Add return statement in `/invocations`. Use `/opt/ml/model` as default `MODEL_DIR` (SageMaker convention).

---

### Tests

#### [NEW] [conftest.py](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/tests/conftest.py)

Shared pytest fixtures: sample DataFrame, temp model directory with mock artifacts.

#### [NEW] [test_train.py](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/tests/training/test_train.py)

Tests for:
- `load_data()` — returns DataFrame with expected columns
- `preprocess()` — output shapes, scaler applied correctly
- `train_model()` — model has `predict` method, returns correct shape
- `evaluate_model()` — returns dict with `mae` and `r2_score`, values in expected ranges
- `save_artifacts()` — creates `model.pkl` and `scaler.pkl` on disk

#### [NEW] [test_inference.py](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/tests/inference/test_inference.py)

Tests for:
- `load_artifacts()` — loads model and scaler from directory
- `predict()` — returns list of floats, correct length
- FastAPI `/ping` endpoint — returns 200
- FastAPI `/invocations` endpoint — returns prediction given valid input

---

### Housekeeping

#### [MODIFY] [requirements.txt](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/requirements.txt)

Add `fastapi`, `uvicorn`, `httpx` (for test client), `sagemaker`, `boto3`.

#### [MODIFY] [.gitignore](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/.gitignore)

Add `models/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`.

#### [DELETE] [Dockerfile.training](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/Dockerfile.training)
#### [DELETE] [Dockerfile.inference](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/Dockerfile.inference)

Empty placeholder files replaced by `docker/training/Dockerfile` and `docker/inference/Dockerfile`.

---

## Things You Didn't Mention But Need

| Missing Piece | Why It Matters |
|---------------|----------------|
| **`GITHUB_TOKEN` secret** | The training container needs to `git clone` your private repo. Add a GitHub PAT as a secret. |
| **`GITHUB_REPO` variable** | The training entrypoint needs `owner/repo` to clone. Passed via `run_training.py`. |
| **`GIT_COMMIT_SHA` in training env** | Without pinning the commit, training runs can't be reproduced. The CI must pass `github.sha` into the SageMaker container. |
| **DVC S3 creds inside container** | The training container needs `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` to `dvc pull` from S3. Passed as env vars. |
| **`conftest.py` + `__init__.py`** | Pytest needs `__init__.py` in test subdirectories and shared fixtures in `conftest.py`. |
| **`requirements_train.txt` / `requirements_inference.txt`** | Separate pip requirements for each container (root `requirements.txt` is for CI runner + tests only). |
| **`httpx` for test client** | FastAPI's `TestClient` requires `httpx` for async testing. |

---

## Verification Plan

### Automated Tests
```bash
# Run all tests locally
pytest tests/ -v

# Run only training tests
pytest tests/training/ -v

# Run only inference tests
pytest tests/inference/ -v
```

### Docker Build Verification
```bash
# Verify training image builds
docker build -f docker/training/Dockerfile -t test-train .

# Verify inference image builds
docker build -f docker/inference/Dockerfile -t test-inference .
```

### Manual Verification
- Push to GitHub → verify CI workflow passes (tests + build + push image + submit training job)
- After training completes → check MLflow Registry has new model version
- Approve model in MLflow UI → trigger CD workflow → verify endpoint is live
