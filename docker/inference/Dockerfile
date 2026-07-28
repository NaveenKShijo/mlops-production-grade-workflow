FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# Install inference dependencies
COPY requirements_inference.txt /tmp/requirements_inference.txt
RUN pip install --no-cache-dir -r /tmp/requirements_inference.txt

# Bake inference source code into the image at build time
# Model artifacts (model.pkl, scaler.pkl) are NOT here —
# SageMaker extracts model.tar.gz and mounts them at /opt/ml/model/ before startup
COPY src/inference/ /opt/ml/code/

COPY docker/inference/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/entrypoint.sh"]