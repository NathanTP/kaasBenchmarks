#!/bin/bash
set -e

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda -f
$HOME/miniconda/bin/conda init
$HOME/miniconda/bin/conda create -y --name kaas python==3.8
echo "conda activate kaas" >> $HOME/.bashrc

# Cuda stuff (follows instructions from https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

sudo apt update
sudo apt install -y mosh cmake llvm-dev build-essential cuda nvidia-cuda-toolkit

echo "WARNING: You must restart your shell before proceeding (to activate anaconda)!"
