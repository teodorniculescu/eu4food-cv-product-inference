#!/bin/bash

ENV_NAME=eu4food_cv_product_inference_venv 

sudo apt update
sudo apt -y install software-properties-common
sudo apt -y jq
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt -y install python3.8 python3.8-venv

python3.8 -m venv $ENV_NAME
source $ENV_NAME/bin/activate
pip3 install --upgrade pip setuptools
pip3 install opencv-contrib-python==4.1.2.30 opencv-python==4.1.2.30 scikit-learn
pip3 install torch torchvision torchaudio
pip3 install scikit-learn
pip3 install seaborn
pip3 install matplotlib
pip3 install tqdm

# TODO add train.sh to crontab
