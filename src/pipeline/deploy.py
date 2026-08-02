"""
deploy.py — CD Script
Reads the Production-approved model from MLflow Registry and deploys it to SageMaker.

Run by CI/CD after a human approves the model in the MLflow UI:
    python src/pipeline/deploy.py

Flow:
    1. Connect to MLflow Registry → find the version of InsuranceChargesModel with alias 'champ-production'
    2. Get its run_id → read the sagemaker_job_name param logged during training
    3. Derive the model.tar.gz S3 URI from sagemaker_job_name
       (SageMaker always stores output at: s3://<bucket>/training-output/<job_name>/output/model.tar.gz)
    4. Deploy (or update) the SageMaker Endpoint with that model URI + inference image
"""

import os
import boto3
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import sagemaker
from sagemaker.model import Model

# ── Config from environment variables (set in GitHub Actions secrets / cd.yaml) ─
MLFLOW_URI = os.environ["MLFLOW_URI"]
IMAGE_URI          = os.environ["IMAGE_URI"]           
AWS_REGION         = os.environ.get("AWS_DEFAULT_REGION", "eu-north-1")
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE", "arn:aws:iam::565265042094:role/MLOps-role")
sagemaker_output_path = os.getenv("SAGEMAKER_OUTPUT_PATH")
ENDPOINT_NAME      = os.getenv("SAGEMAKER_ENDPOINT") 
MODEL_ALIAS = "champ-production"  # simply custom name
REGISTERED_MODEL   = "InsuranceChargesModel"   # specified in train.py 

# ── Step 1: Connect to MLflow and query the Registry ────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)
client = MlflowClient()

print(f"Querying MLflow Registry for Production version of '{REGISTERED_MODEL}'...")

try:
    prod_model = client.get_model_version_by_alias(REGISTERED_MODEL, MODEL_ALIAS)
except MlflowException as e:
    print(e.message)
    print(f"Skipping deployment, due to the exception")    
    exit(1)

run_id       = prod_model.run_id
version_num  = prod_model.version
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



# Here our goal is to obtain the .tar.gz file corresponding to the model that is pushed to production in mlflow registry. 
# For that, we first find the job_name of that selected model. Then we easily track .tar.gz using the job_name, as the sagemaker normally save the .tar.gz package along with job_name by default. 


# ── Step 3: Derive the model.tar.gz S3 URI ───────────────────────────────────────
# SageMaker always stores output at: s3://<output_path>/<job_name>/output/model.tar.gz
model_data_uri = (    
    f"{sagemaker_output_path}{sagemaker_job_name}/output/model.tar.gz"
)
print(f"model.tar.gz location: {model_data_uri}")

# ── Step 4: Deploy or update the SageMaker Endpoint ─────────────────────────────
# The inference image has the FastAPI code baked in.
# SageMaker will extract model.tar.gz → /opt/ml/model/ before the container starts.
inference_image_uri = IMAGE_URI
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
