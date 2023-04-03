#!/bin/bash

sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.8
sudo apt install python3.8-venv

python3.8 -m venv my_virtual_environment/bin/activate
source my_virtual_environment/bin/activate
pip3 install --upgrade pip
pip3 install --upgrade setuptools
pip3 install opencv-contrib-python==4.1.2.30 opencv-python==4.1.2.30 scikit-learn
pip3 install torch torchvision torchaudio
pip3 install tqdm
