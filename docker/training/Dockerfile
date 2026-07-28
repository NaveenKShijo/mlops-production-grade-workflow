FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

# Downloads latest package information -> Downloads and installs git & curl, while ignoring recommended package installs -> Deleting downloaded package cache lists after installation

COPY requirements_train.txt /tmp/requirements_train.txt
RUN pip install --no-cache-dir -r /tmp/requirements_train.txt

COPY docker/training/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh  
# making the entrypoint script executable

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/entrypoint.sh"]
# It tells that 'entrypoint.sh' is the main program that should always run when the container starts