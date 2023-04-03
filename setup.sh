#!/bin/bash

sudo apt update
sudo apt -y install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt -y install python3.8 python3.8-venv

env_name=eu4food_cv_product_inference_venv 

python3.8 -m venv $env_name
source $env_name/bin/activate
pip3 install --upgrade pip setuptools
pip3 install opencv-contrib-python==4.1.2.30 opencv-python==4.1.2.30 scikit-learn
pip3 install torch torchvision torchaudio
pip3 install scikit-learn
pip3 install tqdm

chmod 777 train.sh
# TODO add train.sh to crontab
