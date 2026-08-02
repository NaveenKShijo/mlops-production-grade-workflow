# This is a dedicated Sagemaker launcher script, which is run to submit the training job to Sagemaker

import boto3
from sagemaker.estimator import Estimator 
import os, time


image_uri = os.getenv("IMAGE_URI")
mlflow_uri = os.getenv("MLFLOW_URI")
git_commmit_sha = os.getenv("GIT_COMMIT_SHA")
github_repo = os.getenv("GITHUB_REPO")

# SageMaker configuration details
sagemaker_role = os.getenv("SAGEMAKER_ROLE")
sagemaker_instance_type = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.t3.medium") # ml.m5.large
sagemaker_output_path = os.getenv("SAGEMAKER_OUTPUT_PATH")

# All these env variables will be forwarded to the sagemaker container running train.py 



job_name = f"insurance-training"

estimator = Estimator(    
    image_uri = image_uri,
    role = sagemaker_role, # role ARN
    instance_count = 1, # train using 1 EC2 instance, specify more for distributed training
    instance_type = sagemaker_instance_type,  # ml.m5.large for larger workloads
    output_path = sagemaker_output_path,
    base_job_name = job_name, # base_job_name is a prefix. SageMaker appends a unique suffix to create the real TrainingJobName (eg: insurance-training-1754-2026-08-01-14-23-45-511).
    
    
    # SageMaker injects these env variables into the running container for training
    environment = {
        "GIT_COMMIT_SHA": git_commmit_sha,
        "GITHUB_REPO": github_repo,
        "MLFLOW_URI": mlflow_uri,
        # Instead of hardcoding the MLflow server URL, you pass it from sagemaker. Thus making Docker image reusable. 
        
        # SageMaker auto-injects a TRAINING_JOB_NAME env var into every training container, set to the real, suffixed TrainingJobName.        
    }
)

# estimator2 = Estimator(....)

estimator.fit()  # here using dvc to pull dataset

# estimator2.fit(inputs = "path to dataset") 
# inputs -> to define the path of dataset. Here SageMaker downloads the data before container starts. It makes the image reusable across different datasets


# submits the training job to Sagemaker

