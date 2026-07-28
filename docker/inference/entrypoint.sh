#!/bin/bash

# Shebang: tells Linux to execute this script using the bash shell

set -e  # If any command fails, stop the script immediately

# By this point, SageMaker has already:
#   1. Pulled your image from DockerHub
#   2. Downloaded model.tar.gz from S3
#   3. Extracted model.pkl and scaler.pkl into /opt/ml/model/
# We just need to verify and start the server.

echo "--- Model artifacts available at /opt/ml/model/ ---"
ls -la /opt/ml/model/

echo "--- Starting FastAPI inference server on port 8080 ---"
cd /opt/ml/code

# SageMaker health check: GET /ping  → must return HTTP 200
# SageMaker prediction:   POST /invocations → returns prediction JSON
uvicorn main:app --host 0.0.0.0 --port 8080
