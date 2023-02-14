#!/bin/bash

#some art
sudo apt install figlet
figlet “DRL Trading”

echo "Fuck everything bash script rules"  

#env name
echo "Enter the env name(choose wisely):"  
read env_name
echo "your choice: $env_name"

#python version
echo "Enter the pytohn version you want(prefarable:3.7):"
read python_version

#conda env creation and mamba installtion
conda create -n $env_name python=$python_version
eval "$(conda shell.bash hook)"
conda activate $env_name
conda install -c conda-forge mamba

echo "Pytorch default install:"
mamba install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts

#lib installation 
mamba install -c conda-forge gym
conda install -c conda-forge tensorboard
pip install pandas
pip install numpy
pip install wandb
pip install finta
pip install yfinance
pip install ta
pip install gym-anytrading
pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
figlet “Done”
