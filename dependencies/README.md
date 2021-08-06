# EC2 Setup
AMI: Ubuntu Server 20.04 (x86)
Instance Type:
    - develop: p3.2xlarge
    - experiment: p3.16xlarge
Disk Size: 64GB

## Recommended First Time Setup
Copy over your ssh key for github to the instance and any ssh configs you need.

Update package manager and get basics:

    sudo apt update
    sudo apt install tmux mosh python3.8

Install Anaconda (miniconda):

    wget wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda -f
    ./miniconda/bin/conda init

Clone kaasBenchmarks

    git clone --recurse-submodules git@github.com:NathanTP/kaasBenchmarks.git


