# MLflow Registry + CI/CD Implementation — Full Code Guide

This document shows exactly how the three pieces of your project connect:
1. `train.py` — Logs model + registers to MLflow Registry, logs training job name
2. `deploy.py` (new) — Reads approved model from Registry, finds `.tar.gz`, deploys SageMaker Endpoint
3. `ci.yaml` (updated) — Runs training, then triggers deploy if Registry model is approved

---

## How the Three Files Connect

```
ci.yaml
  │
  ├─► [Step 1] Build & push training Docker image
  ├─► [Step 2] run_training.py → SageMaker Training Job → saves model.tar.gz to S3
  │                                                      → train.py logs run_id + job_name to MLflow
  │                                                      → train.py registers model in MLflow Registry (Pending)
  │
  ├─► [Step 3] deploy.py → queries MLflow Registry for "Production" version
  │                      → gets run_id → gets sagemaker_job_name param
  │                      → derives model.tar.gz S3 URI from job_name
  │                      → deploys SageMaker Endpoint with that URI
  └─► Done
```

> **Note**: The human approval step (changing status from "None" to "Production" in MLflow UI)
> happens BETWEEN training and deployment. The deploy step checks this status and only proceeds
> if a "Production" version exists.

---

## File 1: `src/training/train.py` (update `main()`)

Add two things:
1. Log the SageMaker training job name as an MLflow param so `deploy.py` can trace back to the `.tar.gz`
2. `registered_model_name` is already added — good!

```python
def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(BASE_DIR, "data", "insurance.csv")
    model_dir = os.getenv("SM_MODEL_DIR", os.path.join(BASE_DIR, "models"))

    mlflow_uri = os.getenv("MLFLOW_URI")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("Insurance charges prediction")

    with mlflow.start_run():
        df = load_data(data_path)

        # ✅ KEY: Log the SageMaker training job name so deploy.py can find the model.tar.gz
        # SageMaker injects this env var automatically inside the training container
        sagemaker_job_name = os.getenv("TRAINING_JOB_NAME", "local-run")
        mlflow.log_param("sagemaker_job_name", sagemaker_job_name)

        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        mlflow.log_param("git_commit_hash", commit_hash)
        mlflow.log_param("dataset", data_path)
        mlflow.log_param("num_rows", len(df))
        mlflow.log_param("target_column", "charges")
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("scaler", "MinMaxScaler")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        X_train, X_test, y_train, y_test, scaler = preprocess(df)
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_metrics({"mae": metrics['mae'], "r2_score": metrics['r2_score']})

        # ✅ Save artifacts for SageMaker → gets packaged as model.tar.gz
        save_artifacts(model, scaler, model_dir)
        mlflow.log_artifact(os.path.join(model_dir, "scaler.pkl"))

        print(f"MAE = {metrics['mae']:.2f}")
        print(f"R2 score = {metrics['r2_score']:.4f}")

        # ✅ Log model AND register it in MLflow Model Registry (status: None → pending approval)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_deploy",          # folder name inside MLflow artifact store
            registered_model_name="InsuranceChargesModel",  # creates/increments version in Registry
            pip_requirements=["scikit-learn", "pandas", "numpy"]
        )
```

After training, the MLflow Registry will show:
```
InsuranceChargesModel
  └─ Version 1  │  Stage: None  │  run_id: abc123
  └─ Version 2  │  Stage: None  │  run_id: def456   ← latest training run
```
A human then goes to the MLflow UI and transitions Version 2 → "Production".

---

## File 2: `src/pipeline/deploy.py` (NEW FILE)

This script is the heart of the CD pipeline. It:
1. Connects to MLflow Registry → finds the "Production" approved version
2. Gets its `run_id` → reads `sagemaker_job_name` param from that run
3. Derives the `model.tar.gz` S3 URI from `sagemaker_job_name`
4. Deploys (or updates) the SageMaker Endpoint

```python
"""
deploy.py — CD Script
Reads the Production-approved model from MLflow Registry and deploys it to SageMaker.

Run by CI/CD after a human approves the model in the MLflow UI:
    python src/pipeline/deploy.py
"""

import os
import boto3
import mlflow
from mlflow.tracking import MlflowClient
import sagemaker
from sagemaker.model import Model

# ── Config from environment variables (set in GitHub Actions secrets) ──────────
MLFLOW_URI         = os.environ["MLFLOW_URI"]
DOCKER_USERNAME    = os.environ["DOCKER_USERNAME"]
IMAGE_TAG          = os.environ["IMAGE_TAG"]           # same tag as the inference Docker image
AWS_REGION         = os.environ.get("AWS_REGION", "us-east-1")
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE", "arn:aws:iam::565265042094:role/MLOps-role")
S3_OUTPUT_BUCKET   = "naveen-sagemakeroutput"
ENDPOINT_NAME      = "insurance-charges-endpoint"
REGISTERED_MODEL   = "InsuranceChargesModel"

# ── Step 1: Connect to MLflow and query the Registry ──────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

print(f"Querying MLflow Registry for Production version of '{REGISTERED_MODEL}'...")
prod_versions = client.get_latest_versions(REGISTERED_MODEL, stages=["Production"])

if not prod_versions:
    print("No model in 'Production' stage found. Skipping deployment.")
    print("Go to MLflow UI → Model Registry → Approve a version first.")
    exit(0)

prod_version = prod_versions[0]
run_id       = prod_version.run_id
version_num  = prod_version.version
print(f"Found: {REGISTERED_MODEL} v{version_num}  (run_id={run_id})")

# ── Step 2: Get the SageMaker training job name from that MLflow run ───────────
run = client.get_run(run_id)
sagemaker_job_name = run.data.params.get("sagemaker_job_name")

if not sagemaker_job_name or sagemaker_job_name == "local-run":
    raise ValueError(
        f"Run {run_id} has no valid 'sagemaker_job_name' param. "
        "Was this model trained on SageMaker?"
    )
print(f"Linked SageMaker training job: {sagemaker_job_name}")

# ── Step 3: Derive the model.tar.gz S3 URI from the training job name ──────────
# SageMaker always stores output at: s3://<output_path>/<job_name>/output/model.tar.gz
model_data_uri = f"s3://{S3_OUTPUT_BUCKET}/training-output/{sagemaker_job_name}/output/model.tar.gz"
print(f"model.tar.gz location: {model_data_uri}")

# ── Step 4: Deploy the SageMaker Endpoint ──────────────────────────────────────
inference_image_uri = f"{DOCKER_USERNAME}/ml_inference:{IMAGE_TAG}"

sm_model = Model(
    image_uri=inference_image_uri,     # Your FastAPI inference Docker image
    model_data=model_data_uri,         # The .tar.gz from the approved training run
    role=SAGEMAKER_ROLE_ARN,
    name=f"insurance-model-v{version_num}",
)

# Check if endpoint already exists → update it; otherwise create fresh
sm_client = boto3.client("sagemaker", region_name=AWS_REGION)
existing_endpoints = [e["EndpointName"] for e in sm_client.list_endpoints()["Endpoints"]]

if ENDPOINT_NAME in existing_endpoints:
    print(f"Endpoint '{ENDPOINT_NAME}' exists. Updating...")
    sm_model.deploy(
        initial_instance_count=1,
        instance_type="ml.t3.medium",
        endpoint_name=ENDPOINT_NAME,
        update_endpoint=True,          # zero-downtime rolling update
    )
else:
    print(f"Creating new endpoint '{ENDPOINT_NAME}'...")
    sm_model.deploy(
        initial_instance_count=1,
        instance_type="ml.t3.medium",
        endpoint_name=ENDPOINT_NAME,
    )

print(f"✅ Endpoint '{ENDPOINT_NAME}' is live with model v{version_num} (run_id={run_id})")
```

---

## File 3: `.github/workflows/ci.yaml` (Updated)

Split into two workflows:

### `ci.yaml` — Runs on every push (CI + Training)
```yaml
name: CI and SageMaker Training
run-name: ${{ github.actor }} triggered CI + Training
on: [push]

jobs:
  ci_and_train:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: pytest

      - name: Login to DockerHub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      # Build and push TRAINING image
      - name: Build training Docker image
        run: docker build -f Dockerfile.train -t ${{ secrets.DOCKER_USERNAME }}/ml_train:${{ github.sha }} .

      - name: Push training image to DockerHub
        run: docker push ${{ secrets.DOCKER_USERNAME }}/ml_train:${{ github.sha }}

      # Submit SageMaker training job
      # train.py inside the container will:
      #   1. save_artifacts() → model.tar.gz on S3
      #   2. log sagemaker_job_name to MLflow
      #   3. register model in MLflow Registry (status: None — pending approval)
      - name: Submit SageMaker training job
        env:
          IMAGE_TAG: ${{ github.sha }}
          MLFLOW_URI: ${{ secrets.MLFLOW_URI }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: python src/pipeline/run_training.py
```

### `cd.yaml` (NEW) — Triggered manually after human approves model in MLflow UI
```yaml
name: CD - Deploy Approved Model to SageMaker
on:
  workflow_dispatch:     # Manually triggered from GitHub Actions UI after approval
    inputs:
      image_tag:
        description: "Inference Docker image tag to deploy"
        required: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Login to DockerHub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      # Build and push INFERENCE image
      - name: Build inference Docker image
        run: docker build -f Dockerfile.inference -t ${{ secrets.DOCKER_USERNAME }}/ml_inference:${{ github.event.inputs.image_tag }} .

      - name: Push inference image to DockerHub
        run: docker push ${{ secrets.DOCKER_USERNAME }}/ml_inference:${{ github.event.inputs.image_tag }}

      # Deploy: reads approved model from Registry → finds model.tar.gz → creates/updates endpoint
      - name: Deploy Production model to SageMaker Endpoint
        env:
          MLFLOW_URI: ${{ secrets.MLFLOW_URI }}
          DOCKER_USERNAME: ${{ secrets.DOCKER_USERNAME }}
          IMAGE_TAG: ${{ github.event.inputs.image_tag }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: us-east-1
          SAGEMAKER_ROLE: ${{ secrets.SAGEMAKER_ROLE_ARN }}
        run: python src/pipeline/deploy.py
```

---

## Complete Data Flow Summary

```
[Push to GitHub]
      │
      ▼
[ci.yaml]
  ├─ pytest ✅
  ├─ Build + push ml_train Docker image
  └─ run_training.py → SageMaker Training Job
                           └─ train.py runs inside Container A
                               ├─ save_artifacts() → /opt/ml/model/
                               │       └─ SageMaker packages → s3://naveen-sagemakeroutput/
                               │                               training-output/<job-name>/
                               │                               output/model.tar.gz
                               └─ mlflow.sklearn.log_model(registered_model_name=...)
                                       ├─ Logs run_id, sagemaker_job_name, metrics to MLflow
                                       └─ Creates: InsuranceChargesModel v2 [Stage: None]

[Human Action — MLflow UI]
  └─ Reviews metrics → Transitions v2 → "Production"

[Manual trigger: cd.yaml]
  ├─ Build + push ml_inference Docker image
  └─ deploy.py
       ├─ Queries MLflow Registry → gets v2 (Production), run_id = "abc123"
       ├─ Reads MLflow run params → sagemaker_job_name = "training-job-abc"
       ├─ Builds S3 URI: s3://naveen-sagemakeroutput/training-output/training-job-abc/output/model.tar.gz
       └─ Deploys SageMaker Endpoint "insurance-charges-endpoint"
               └─ SageMaker extracts model.tar.gz → /opt/ml/model/
                       └─ FastAPI starts → joblib.load("model.pkl"), joblib.load("scaler.pkl")
                               └─ /invocations ready to serve predictions ✅
```
