#!/bin/bash

# this is shebang(# is immediately followed by '!' without a gap). It tells linux to execute the script using bash shell

set -e  # if any command fails, stop the script immediately

echo "---- Cloning repository at commit ${GIT_COMMIT_SHA:-HEAD} ----"
# ${variable:-default_value}  

git clone https://github.com/${GITHUB_REPO}.git /opt/ml/code  # GITHUB_REPO = NaveenKShijo/repo-name
# git clone https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git /opt/ml/code  # for private repository

cd /opt/ml/code

if [ -n "${GIT_COMMIT_SHA}" ]; then
    git checkout "${GIT_COMMIT_SHA}"
fi

echo "--- Pulling dataset via DVC ---"
dvc pull data/insurance.csv
# data/insurance.csv specified to deny DVC from downloading other datasets whose .dvc files are also preesent

echo "--- Start training ---"
python src/training/train.py

