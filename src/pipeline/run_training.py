# This is a dedicated Sagemaker launcher script, which is run to submit the training job to Sagemaker

import boto3
from sagemaker.estimator import Estimator 
import os, time

docker_username = os.getenv("DOCKER_USERNAME")
image_tag = os.getenv("IMAGE_TAG")
mlflow_uri = os.getenv("MLFLOW_URI")
git_commmit_sha = os.getenv("GIT_COMMIT_SHA")
github_repo = os.getenv("GITHUB_REPO")

# SageMaker configuration details
sagemaker_role = os.getenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::565265042094:role/MLOps-role")
sagemaker_instance_type = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.t3.medium")
sagemaker_output_path = os.getenv("SAGEMAKER_OUTPUT_PATH", "s3://naveen-sagemakeroutput/training-output/")

job_name = f"insurance-training-{int(time.time())}"

estimator = Estimator(    
    image_uri = f"{docker_username}/ml_train:{image_tag}",
    role = sagemaker_role, # role ARN
    instance_count = 1, # train using 1 EC2 instance, specify more for distributed training
    instance_type = sagemaker_instance_type,  # ml.m5.large for larger workloads
    output_path = sagemaker_output_path,
    base_job_name = job_name, # tells Sagemaker what to name this training run. Sagemaker will append base_job_name to the output path
    
    
    # SageMaker injects these env variables into the running container for training
    environment = {
        "GIT_COMMIT_SHA": git_commmit_sha,
        "GITHUB_REPO": github_repo,
        "MLFLOW_URI": mlflow_uri,
        # Instead of hardcoding the MLflow server URL, you pass it from sagemaker. Thus making Docker image reusable. 
        "TRAINING_JOB_NAME": job_name
    }
)

# estimator2 = Estimator(....)

estimator.fit()  # here using dvc to pull dataset

# estimator2.fit(inputs = "path to dataset") 
# inputs -> to define the path of dataset. Here SageMaker downloads the data before container starts. It makes the image reusable across different datasets


# submits the training job to Sagemaker

