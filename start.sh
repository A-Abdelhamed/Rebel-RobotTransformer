#!/bin/bash

# Stop the script if any command fails
set -e

echo "--- Starting Setup ---"

# 1. Install System Dependencies
apt-get update
apt-get install -y openssh-server unzip apt-transport-https ca-certificates gnupg curl

# 2. Setup SSH
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
mkdir -p /run/sshd
/usr/sbin/sshd
echo "SSH Server started..."

# 3. Install Python Dependencies
pip install gdown
pip install "jax[cuda12_pip]==0.4.38" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
pip install gsutil
pip install kaggle

# 4. Install Google Cloud SDK
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get update
apt-get install -y google-cloud-cli

# 5. Download and Unzip Data
# (Only download if the folder doesn't exist yet to save time on restarts)
if [ ! -d "/app/IgusToolPicking" ]; then
    echo "Downloading Igus dataset..."
	export KAGGLE_USERNAME=ahmedkzabdelhamed
	export KAGGLE_KEY=KGAT_b6f3c0e66de27c534a9d69d8a1ffafca
    python DownloadDataset.py
    unzip /app/IgusToolPicking.zip -d /app/IgusToolPicking
    rm /app/IgusToolPicking.zip
else
    echo "Dataset already exists. Skipping download."
fi

# 6. Download RT-1-X Model
if [ ! -d "rt_1_x_jax" ]; then
    echo "Downloading RT-1-X model..."
    gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_jax .
fi

echo "--- Setup Complete. Starting Training ---"

# 7. RUN YOUR MAIN SCRIPT
# This replaces the python command you were trying to run manually
python TrainingCode.py