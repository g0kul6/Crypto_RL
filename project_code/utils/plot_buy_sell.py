import os
import sys
from datetime import date
now = date.today()
path_dir = os.path.abspath(os.getcwd())
folder_name = "/CHECKPOINT/checkpoint{}/".format(now)
images_dir = path_dir + folder_name + "point_images/"
test_dir = path_dir + folder_name + "test/"
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch 

import argparse
parser = argparse.ArgumentParser()
#algo
parser.add_argument("--algo",type=str,default="vpg",required=True)
#feature extractor
parser.add_argument("--feature_extractor",type=str,default="mlp",required=True)
#reward type
parser.add_argument("--reward_type",type=str,default="sr",required=True)
args = parser.parse_args()

def plot_action_state(action_path,state_path,rewards_path,cumulative_rewards_path):
    actions = np.load(action_path)
    states = np.load(state_path)
    rewards = np.load(rewards_path)
    cumulative_rewards = np.load(cumulative_rewards_path)
    sell_count = 0
    sells = []
    index_sells = []
    buy_count = 0
    buys = []
    index_buys = []
    plt.plot(range(len(states[1001:2001,0,0])),states[1001:2001,0,0])
    for i in range(len(actions[1000:2000])):
        if actions[i] == 0:
            buy_count += 1
            index_buys.append(i)
            buys.append(states[1001+i,0,0])
        elif actions[i] == 1: 
            sell_count += 1
            index_sells.append(i)
            sells.append(states[1001+i,0,0])
    plt.scatter(index_buys,buys,color="r",label="buy")
    plt.scatter(index_sells,sells,color="b",label="sell")
    plt.title('buy:{},sell:{}'.format(buy_count,sell_count))
    plt.legend()
    plt.savefig("{}{}_{}_{}.png".format(images_dir,args.algo,args.feature_extractor,args.reward_type))
    plt.show()
    print("buy:",buy_count,"sell:",sell_count)
    
    
    
if __name__== '__main__':
    plot_action_state(action_path=test_dir+"action_{}_{}_{}.npy".format(args.algo,args.feature_extractor,args.reward_type),
                    state_path=test_dir+"state_{}_{}_{}.npy".format(args.algo,args.feature_extractor,args.reward_type),
                    rewards_path=test_dir+"rewards_{}_{}_{}.npy".format(args.algo,args.feature_extractor,args.reward_type),
                    cumulative_rewards_path=test_dir+"cumulative_rewards_{}_{}_{}.npy".format(args.algo,args.feature_extractor,args.reward_type))
