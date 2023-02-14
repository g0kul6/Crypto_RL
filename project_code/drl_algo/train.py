import os
import sys
path_dir = os.path.abspath(os.getcwd())
sys.path.append(path_dir+"/project_code")
data_dir = path_dir + "/dataset/BITCOIN/"

import torch
from project_code.utils.cryptoenv_buy_sell_hold import Environment
from utils.data import data_loader
from utils.config import list_indicators,window_size,actor_lr,critic_lr,gamma,max_episodes
from train_util import train_a2c_lstm,train_a2c_mlp,train_a2c_transformer, train_ppo_lstm,train_vpg_lstm,train_vpg_transformer,train_vpg_mlp,train_dqn_lstm,train_dqn_mlp,train_ddqn_lstm,train_ppo_transformer

"""
 get hyper-parmaeters from userr
 1.LR - ACTOR and CRITIC
 2.GAMMA - discount 
 3.EPSILON - epsilon-greedy policy
 4.REWARD TYPE - profit/sr
 5.ALGO - VPG,A2C,PPO,DQN,DDQN
 6.FEATURE EXTRACTOR - MLP,LSTM,TRANSFORMER
"""
import argparse
parser = argparse.ArgumentParser()
#algo
parser.add_argument("--algo",type=str,default="vpg",required=True)
#feature extractor
parser.add_argument("--feature_extractor",type=str,default="mlp",required=True)
#reward type
parser.add_argument("--reward_type",type=str,default="sr",required=True)
#actor and critic lr
parser.add_argument("--lr",type=float,default=actor_lr,required=True)
#gamma
parser.add_argument("--gamma",type=float,default=gamma,required=True)
#device
parser.add_argument("--device",type=str,default="cuda",required=True)
#episodes
parser.add_argument("--epochs",type=int,default=max_episodes,required=True)
args = parser.parse_args()
#env
df_train,df_test=data_loader(path=data_dir + "Gemini_BTCUSD_1h.csv",train_test_split=0.9)
if args.feature_extractor == "transformer" :
    env_train=Environment(df_train,reward=args.reward_type,state_space=1+len(list_indicators),tech_indicators=list_indicators,ws=512,start_point=512,end_point=len(df_train)-1,random=True,device=args.device,env_type="train")
    env_test=Environment(df_test,reward=args.reward_type,state_space=1+len(list_indicators),tech_indicators=list_indicators,ws=512,start_point=512,end_point=len(df_test)-1,random=False,device=args.device)
else:
    env_train=Environment(df_train,reward=args.reward_type,state_space=1+len(list_indicators),tech_indicators=list_indicators,ws=window_size,start_point=24,end_point=len(df_train),random=True,device=args.device,env_type="train")
    env_test=Environment(df_test,reward=args.reward_type,state_space=1+len(list_indicators),tech_indicators=list_indicators,ws=window_size,start_point=24,end_point=len(df_test),random=False,device=args.device)

#train
if args.algo == "vpg":
    if args.feature_extractor == "mlp":
        train_vpg_mlp(env=env_train,lr=args.lr,gamma=args.gamma,device=args.device,reward_type=args.reward_type,epoch=args.epochs)
    elif args.feature_extractor == "lstm":
        train_vpg_lstm(env=env_train,lr=args.lr,gamma=args.gamma,device=args.device,reward_type=args.reward_type,epoch=args.epochs)
    elif args.feature_extractor == "transformer":
        train_vpg_transformer(env=env_train,lr=args.lr,gamma=args.gamma,device=args.device,reward_type=args.reward_type,epoch=args.epochs)
elif args.algo == "dqn":
    if args.feature_extractor == "mlp":
        train_dqn_mlp(env=env_train,lr=args.lr,total_timesteps=args.epochs,target_network_frequency=100,batch_size=32,gamma=args.gamma,epsilon_decay=9.9e-6,device=args.device,reward_type=args.reward_type)
    elif args.feature_extractor == "lstm":
        train_dqn_lstm(env=env_train,lr=args.lr,total_timesteps=args.epochs,target_network_frequency=100,batch_size=32,gamma=args.gamma,epsilon_decay=9.9e-6,device=args.device,reward_type=args.reward_type)
elif args.algo == "a2c":
    if args.feature_extractor == "mlp":
        train_a2c_mlp(env=env_train,lr=args.lr,gamma=args.gamma,device=args.device,reward_type=args.reward_type,epoch=args.epochs)
    elif args.feature_extractor == "lstm":
        train_a2c_lstm(env=env_train,lr=args.lr,gamma=args.gamma,device=args.device,reward_type=args.reward_type,epoch=args.epochs)
    elif args.feature_extractor == "transformer":
        train_a2c_transformer(env=env_train,lr=args.lr,gamma=args.gamma,device=args.device,reward_type=args.reward_type,epoch=args.epochs)
elif args.algo == "ppo":
    if args.feature_extractor == "mlp":
        pass
    elif args.feature_extractor == "lstm":
        train_ppo_lstm(env=env_train,state_dim=env_train.observation_space.shape[0],action_dim=env_train.action_space.n,max_episodes=args.epochs,update_timestep=20,lr=args.lr,K_epochs=4,betas=[0.9, 0.990],gamma=args.gamma,eps_clip=0.2,reward_type=args.reward_type,device=args.device)
    elif args.feature_extractor == "transformer":
        train_ppo_transformer(env=env_train,state_dim=env_train.observation_space.shape[0],action_dim=env_train.action_space.n,max_episodes=args.epochs,update_timestep=20,lr=args.lr,K_epochs=4,betas=[0.9, 0.990],gamma=args.gamma,eps_clip=0.2,reward_type=args.reward_type,device=args.device)
elif args.algo == "ddqn":
    if args.feature_extractor == "mlp":
        pass
    elif args.feature_extractor == "lstm":
        train_ddqn_lstm(env=env_train,lr=args.lr,total_timesteps=args.epochs,target_network_frequency=100,batch_size=32,gamma=args.gamma,epsilon_decay=9.9e-6,device=args.device,reward_type=args.reward_type)




  
