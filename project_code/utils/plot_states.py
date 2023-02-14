import os
import sys
path_dir = os.path.abspath(os.getcwd())
data_dir = path_dir + "/dataset/BITCOIN/"

import torch
import wandb
import numpy as np
from data import data_loader


import argparse
parser = argparse.ArgumentParser()
#mode
parser.add_argument("--mode",type=str,default="train",required=True)
#wandb project name
parser.add_argument("--project_name",type=str,default="anything",required=True)
args = parser.parse_args()

#data
df_train,df_test=data_loader(path=data_dir + "Gemini_BTCUSD_1h.csv",train_test_split=0.9)

def plot_states(mode):
    wandb.init(project=args.project_name,name=mode+"_states",entity="g0kul6")
    if mode == "train":
        data = df_train
    elif mode == "test":
        data = df_test
    for i in range(len(data["Close"])):
        wandb.log({"Close":data.iloc[i]["Close"],"RSI":data.iloc[i]["Unorm_rsi"]})

if __name__== '__main__':
    plot_states(mode=args.mode)


