# AWS SageMaker ML Lifecycle — Implementation Plan

## Background

Your project trains a `LinearRegression` model on `insurance.csv` and exposes predictions via a FastAPI service. Right now, code, data, and CI are set up for a basic Docker + DockerHub flow. The goal is to migrate this to **AWS SageMaker** for both training (SageMaker Training Jobs) and inference (SageMaker Endpoints), while keeping code changes **minimal**.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  TRAINING FLOW                                                       │
│                                                                      │
│  GitHub Push → CI builds base training image → pushes to ECR        │
│       ↓                                                              │
│  SageMaker Training Job (spins up the base image)                   │
│       ↓  [container startup]                                        │
│  entrypoint.sh: git clone repo + dvc pull dataset                   │
│       ↓                                                              │
│  python src/training/train.py  →  saves model.pkl to S3 (/opt/ml/model) │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  INFERENCE FLOW                                                      │
│                                                                      │
│  GitHub Push → CI builds inference image → pushes to ECR            │
│       ↓                                                              │
│  SageMaker Endpoint (spins up the inference image)                  │
│       ↓  [container startup]                                        │
│  entrypoint.sh: pulls model.pkl + scaler.pkl from S3 model registry │
│       ↓                                                              │
│  FastAPI serves /invocations and /ping (SageMaker contract)         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Open Questions

> [!IMPORTANT]
> **Where is your dataset stored for DVC pull?**  
> Your `data/insurance.csv.dvc` pointer exists, but the DVC remote storage backend is not configured in `.dvc/config`. For the training container to `dvc pull` the dataset at runtime, it needs credentials and a remote URL (e.g., S3 bucket). Please confirm:
> - Will you use **S3 as DVC remote storage** (recommended since you're already on AWS)?
> - Or will you download the CSV directly from S3 (simpler, bypassing DVC)?

> [!IMPORTANT]
> **Where will trained models be stored?**
> SageMaker saves model artifacts to S3 automatically (at `s3://<bucket>/output/model.tar.gz`). For the inference container to pull the model at startup, it needs an S3 path.  
> - Do you have an S3 bucket name in mind? (e.g., `my-insurance-ml-bucket`)

> [!IMPORTANT]
> **ECR Repository names?**  
> AWS ECR (Elastic Container Registry) will replace DockerHub for storing images. Do you have existing ECR repos, or should I include the ECR repo creation step in CI?

> [!NOTE]
> The plan below assumes S3 for both DVC data storage and model registry. Placeholder values like `YOUR_S3_BUCKET` and `YOUR_AWS_REGION` will need to be replaced with your actual values.

---

## Proposed Changes

### 1. `src/training/train.py` — Minimal change

#### [MODIFY] [train.py](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/src/training/train.py)

**What changes:** SageMaker mounts artifact output at `/opt/ml/model`. The `MODEL_DIR` env var already handles this — **no code change needed**. However, the `mlflow.set_tracking_uri()` must be configured to point to a remote MLflow server or S3-backed store (or be disabled) since the container won't have the local `mlflow.db`.

**Change:** Add a fallback tracking URI:
```python
# In main(), before mlflow.set_experiment(...)
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(mlflow_uri)
```

---

### 2. `src/inference/main.py` — SageMaker contract

#### [MODIFY] [main.py](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/src/inference/main.py)

**What changes:** SageMaker's real-time inference endpoint expects two specific HTTP routes:
- `GET /ping` → health check, must return HTTP 200
- `POST /invocations` → actual prediction endpoint

Your current `/home` route won't satisfy SageMaker. You need to add `/ping` and `/invocations`.

**Change:** Replace `/home` with SageMaker-compatible routes and wire in prediction logic:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import pandas as pd
from predict import load_artifacts, predict

app = FastAPI()

MODEL_DIR = os.getenv("MODEL_DIR", "/opt/ml/model")
model, scaler = load_artifacts(MODEL_DIR)

@app.get("/ping")
def ping():
    """SageMaker health check — must return 200"""
    return JSONResponse(status_code=200, content={"status": "healthy"})

@app.post("/invocations")
async def invocations(request: Request):
    """SageMaker prediction endpoint"""
    body = await request.json()
    input_df = pd.DataFrame([body])
    result = predict(model, scaler, input_df)
    return JSONResponse(content={"prediction": result})
```

---

### 3. `src/inference/predict.py` — No change needed

[predict.py](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/src/inference/predict.py) already reads `MODEL_DIR` from the environment. SageMaker will set this to `/opt/ml/model`. ✅ No changes required.

---

### 4. New file: `docker/train/Dockerfile` — Training base image

#### [NEW] `docker/train/Dockerfile`

This image contains **only the Python environment**. Source code and data are fetched at runtime via `entrypoint.sh`.

```dockerfile
FROM python:3.11-slim

# Install system tools needed at runtime
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install AWS CLI (for S3 access / dvc pull)
RUN pip install --no-cache-dir awscli

# Install Python dependencies
COPY requirements_train.txt /tmp/requirements_train.txt
RUN pip install --no-cache-dir -r /tmp/requirements_train.txt

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# SageMaker sets SAGEMAKER_PROGRAM, but we use our own entrypoint
ENTRYPOINT ["/entrypoint.sh"]
```

#### [NEW] `docker/train/entrypoint.sh`

This script runs when the container starts. It clones the repo and pulls the dataset dynamically:

```bash
#!/bin/bash
set -e

# 1. Clone the source code from GitHub
git clone $GITHUB_REPO_URL /opt/ml/code
cd /opt/ml/code

# 2. Pull dataset from S3 (two options — pick one):
#    Option A: Direct S3 copy (simpler)
aws s3 cp s3://$S3_BUCKET/data/insurance.csv data/insurance.csv

#    Option B: DVC pull (keeps versioning)
# dvc remote add -d s3remote s3://$S3_BUCKET/dvc-storage --no-default
# dvc pull data/insurance.csv

# 3. Run training
#    MODEL_DIR=/opt/ml/model is where SageMaker expects output artifacts
export MODEL_DIR=/opt/ml/model
export DATA_PATH=/opt/ml/code/data/insurance.csv
python src/training/train.py
```

#### [NEW] `docker/train/requirements_train.txt`

Separated from the main `requirements.txt` so the inference image stays lean:

```
numpy
pandas
matplotlib
joblib
scikit-learn
mlflow
dvc[s3]
```

---

### 5. New file: `docker/inference/Dockerfile` — Inference image

#### [NEW] `docker/inference/Dockerfile`

This image pulls the trained model from S3 (model registry) at container startup:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install AWS CLI for model download
RUN pip install --no-cache-dir awscli

# Install inference dependencies
COPY requirements_inference.txt /tmp/requirements_inference.txt
RUN pip install --no-cache-dir -r /tmp/requirements_inference.txt

# Copy inference source code (baked into image)
COPY src/inference/ /opt/ml/code/

# Copy entrypoint
COPY docker/inference/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
```

#### [NEW] `docker/inference/entrypoint.sh`

```bash
#!/bin/bash
set -e

# 1. Download model artifacts from S3 model registry
mkdir -p /opt/ml/model
aws s3 cp s3://$S3_BUCKET/models/model.pkl /opt/ml/model/model.pkl
aws s3 cp s3://$S3_BUCKET/models/scaler.pkl /opt/ml/model/scaler.pkl

# 2. Start FastAPI server
#    SageMaker expects the server to listen on port 8080
export MODEL_DIR=/opt/ml/model
cd /opt/ml/code
uvicorn main:app --host 0.0.0.0 --port 8080
```

#### [NEW] `docker/inference/requirements_inference.txt`

```
numpy
pandas
joblib
scikit-learn
fastapi
uvicorn
```

---

### 6. `requirements.txt` — No change

The root `requirements.txt` is used by CI for running `pytest`. It stays as-is.

---

### 7. `.github/workflows/ci.yaml` — Replace with SageMaker-ready CI/CD

#### [MODIFY] [ci.yaml](file:///c:/Users/Naveen/Desktop/DataScience/Week41/LProj1/.github/workflows/ci.yaml)

The current workflow has issues (`python-version` format wrong, no image tagging strategy, DockerHub instead of ECR). The new workflow:

```yaml
name: ML CI/CD Pipeline

on:
  push:
    branches: [main]

env:
  AWS_REGION: YOUR_AWS_REGION           # e.g. us-east-1
  ECR_REGISTRY: YOUR_AWS_ACCOUNT_ID.dkr.ecr.YOUR_AWS_REGION.amazonaws.com
  TRAIN_IMAGE_NAME: insurance-train
  INFERENCE_IMAGE_NAME: insurance-inference

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"           # ← Fix: was "python:3.11" (wrong format)
      - run: pip install -r requirements.txt
      - run: pytest

  build-and-push-train:
    name: Build & Push Training Image
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build & push training image
        run: |
          IMAGE_TAG=${{ github.sha }}
          docker build \
            -f docker/train/Dockerfile \
            -t $ECR_REGISTRY/$TRAIN_IMAGE_NAME:$IMAGE_TAG \
            -t $ECR_REGISTRY/$TRAIN_IMAGE_NAME:latest \
            .
          docker push $ECR_REGISTRY/$TRAIN_IMAGE_NAME:$IMAGE_TAG
          docker push $ECR_REGISTRY/$TRAIN_IMAGE_NAME:latest

  build-and-push-inference:
    name: Build & Push Inference Image
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build & push inference image
        run: |
          IMAGE_TAG=${{ github.sha }}
          docker build \
            -f docker/inference/Dockerfile \
            -t $ECR_REGISTRY/$INFERENCE_IMAGE_NAME:$IMAGE_TAG \
            -t $ECR_REGISTRY/$INFERENCE_IMAGE_NAME:latest \
            .
          docker push $ECR_REGISTRY/$INFERENCE_IMAGE_NAME:$IMAGE_TAG
          docker push $ECR_REGISTRY/$INFERENCE_IMAGE_NAME:latest
```

> [!NOTE]
> GitHub Secrets needed: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`. Remove old `DOCKER_USERNAME` / `DOCKER_PASSWORD` secrets.

---

### 8. Final File Structure After Changes

```
LProj1/
├── .github/
│   └── workflows/
│       └── ci.yaml                    ← MODIFIED
├── docker/
│   ├── train/
│   │   ├── Dockerfile                 ← NEW
│   │   ├── entrypoint.sh              ← NEW
│   │   └── requirements_train.txt     ← NEW
│   └── inference/
│       ├── Dockerfile                 ← NEW
│       ├── entrypoint.sh              ← NEW
│       └── requirements_inference.txt ← NEW
├── src/
│   ├── training/
│   │   └── train.py                   ← MINOR CHANGE (add mlflow URI env var)
│   └── inference/
│       ├── main.py                    ← MODIFIED (add /ping + /invocations)
│       └── predict.py                 ← NO CHANGE ✅
├── data/
│   └── insurance.csv.dvc              ← NO CHANGE ✅
└── requirements.txt                   ← NO CHANGE ✅
```

---

## Verification Plan

### After implementing changes

1. **Local Docker build test** (before pushing):
   ```bash
   # Test training image builds
   docker build -f docker/train/Dockerfile -t test-train .

   # Test inference image builds
   docker build -f docker/inference/Dockerfile -t test-inference .
   ```

2. **Local inference test** (mock model dir):
   ```bash
   docker run -e MODEL_DIR=/opt/ml/model \
     -e S3_BUCKET=your-bucket \
     -p 8080:8080 test-inference
   # Hit http://localhost:8080/ping → should return {"status":"healthy"}
   ```

3. **SageMaker Training Job**: Trigger a training job from the AWS console or SDK using the ECR training image URI.

4. **SageMaker Endpoint**: Deploy inference image as a SageMaker real-time endpoint and call `/invocations` with a sample JSON payload.
