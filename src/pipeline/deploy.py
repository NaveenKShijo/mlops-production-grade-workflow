"""
deploy.py — CD Script
Reads the Production-approved model from MLflow Registry and deploys it to SageMaker.

Run by CI/CD after a human approves the model in the MLflow UI:
    python src/pipeline/deploy.py

Flow:
    1. Connect to MLflow Registry → find the "Production" version of InsuranceChargesModel
    2. Get its run_id → read the sagemaker_job_name param logged during training
    3. Derive the model.tar.gz S3 URI from sagemaker_job_name
       (SageMaker always stores output at: s3://<bucket>/training-output/<job_name>/output/model.tar.gz)
    4. Deploy (or update) the SageMaker Endpoint with that model URI + inference image
"""

import os
import boto3
import mlflow
from mlflow.tracking import MlflowClient
import sagemaker
from sagemaker.model import Model

# ── Config from environment variables (set in GitHub Actions secrets / cd.yaml) ─
MLFLOW_URI         = os.environ["MLFLOW_URI"]
DOCKER_USERNAME    = os.environ["DOCKER_USERNAME"]
IMAGE_TAG          = os.environ["IMAGE_TAG"]           # git sha of the inference image
AWS_REGION         = os.environ.get("AWS_DEFAULT_REGION", "us-north-1")
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE", "arn:aws:iam::565265042094:role/MLOps-role")
S3_OUTPUT_BUCKET   = os.getenv("S3_OUTPUT_BUCKET")  
ENDPOINT_NAME      = os.getenv("SAGEMAKER_ENDPOINT") #"insurance-charges-endpoint"
REGISTERED_MODEL   = "InsuranceChargesModel"   # specified in train.py 

# ── Step 1: Connect to MLflow and query the Registry ────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

print(f"Querying MLflow Registry for Production version of '{REGISTERED_MODEL}'...")
prod_versions = client.get_latest_versions(REGISTERED_MODEL, stages=["Production"])

if not prod_versions:
    print("No model found in 'Production' stage. Skipping deployment.")
    print("Go to MLflow UI → Model Registry → transition a version to 'Production' first.")
    exit(0)

prod_version = prod_versions[0]
run_id       = prod_version.run_id
version_num  = prod_version.version
print(f"Found: {REGISTERED_MODEL} v{version_num}  (run_id={run_id})")

# ── Step 2: Get the SageMaker training job name from that MLflow run ─────────────
# train.py logs this at training time: mlflow.log_param("sagemaker_job_name", ...)
run = client.get_run(run_id)
sagemaker_job_name = run.data.params.get("sagemaker_job_name")

if not sagemaker_job_name or sagemaker_job_name == "local-run":
    raise ValueError(
        f"Run {run_id} has no valid 'sagemaker_job_name' param. "
        "Was this model trained on SageMaker? Local runs cannot be deployed."
    )
print(f"Linked SageMaker training job: {sagemaker_job_name}")

# ── Step 3: Derive the model.tar.gz S3 URI ───────────────────────────────────────
# SageMaker always stores output at: s3://<output_path>/<job_name>/output/model.tar.gz
model_data_uri = (
    f"s3://{S3_OUTPUT_BUCKET}/training-output/{sagemaker_job_name}/output/model.tar.gz"
)
print(f"model.tar.gz location: {model_data_uri}")

# ── Step 4: Deploy or update the SageMaker Endpoint ─────────────────────────────
# The inference image has the FastAPI code baked in.
# SageMaker will extract model.tar.gz → /opt/ml/model/ before the container starts.
inference_image_uri = f"{DOCKER_USERNAME}/ml-inference:{IMAGE_TAG}"
print(f"Inference image: {inference_image_uri}")

sm_model = Model(
    image_uri=inference_image_uri,     # FastAPI inference Docker image (code baked in)
    model_data=model_data_uri,         # The .tar.gz SageMaker will mount at /opt/ml/model/
    role=SAGEMAKER_ROLE_ARN,
    name=f"insurance-model-v{version_num}",
)

# Check if the endpoint already exists → update it; otherwise create a fresh one
sm_client = boto3.client("sagemaker", region_name=AWS_REGION)
existing_endpoints = [e["EndpointName"] for e in sm_client.list_endpoints()["Endpoints"]]

if ENDPOINT_NAME in existing_endpoints:
    print(f"Endpoint '{ENDPOINT_NAME}' exists. Performing rolling update...")
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

print(f"Endpoint '{ENDPOINT_NAME}' is live.")
print(f"Model:   {REGISTERED_MODEL} v{version_num}")
print(f"run_id:  {run_id}")
print(f"Image:   {inference_image_uri}")
