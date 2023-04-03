#!/bin/bash

sudo apt update
sudo apt -y install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt -y install python3.8
sudo apt -y install python3.8-venv

$env_name=eu4food_cv_product_inference_venv 

python3.8 -m venv $env_name/bin/activate
source $env_name/bin/activate
pip3 install -y --upgrade pip
pip3 install -y --upgrade setuptools
pip3 install -y opencv-contrib-python==4.1.2.30 opencv-python==4.1.2.30 scikit-learn
pip3 install -y torch torchvision torchaudio
pip3 install -y tqdm
