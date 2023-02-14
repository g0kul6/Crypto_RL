
import os
import sys
from datetime import date
now = date.today()
path_dir = os.path.abspath(os.getcwd())
folder_name = "/CHECKPOINT/checkpoint{}/".format(now)
sys.path.append(path_dir+"/project_code")
data_dir = path_dir + "/dataset/BITCOIN/"
test_dir = path_dir+folder_name+"test/"


import torch
import wandb
import numpy as np
from project_code.utils.cryptoenv_buy_sell_hold import Environment
from utils.data import data_loader
from utils.config import list_indicators,window_size
from drl_algo.models import A2C_LSTM,A2C_MLP,DQN_LSTM,DQN_MLP,VPG_LSTM,VPG_MLP,PPO_LSTM,VPG_TRANSFORMER,A2C_TRANSFORMER,PPO_TRANSFORMER
from drl_algo.memory import Memory_PPO

import argparse
parser = argparse.ArgumentParser()
#algo
parser.add_argument("--algo",type=str,default="vpg",required=True)
#feature extractor
parser.add_argument("--feature_extractor",type=str,default="mlp",required=True)
#reward type
parser.add_argument("--reward_type",type=str,default="sr",required=True)
#device
parser.add_argument("--device",type=str,default="cuda",required=True)

args = parser.parse_args()

#env
df_train,df_test=data_loader(path=data_dir + "Gemini_BTCUSD_1h.csv",train_test_split=0.9)
if args.feature_extractor == "transformer" :
    env_train=Environment(df_train,reward=args.reward_type,state_space=1+len(list_indicators),tech_indicators=list_indicators,ws=512,start_point=512,end_point=len(df_train)-1,random=False,device=args.device,env_type="test")
    env_test=Environment(df_test,reward=args.reward_type,state_space=1+len(list_indicators),tech_indicators=list_indicators,ws=512,start_point=512,end_point=len(df_test)-1,random=False,device=args.device,env_type="test")
else:
    env_train=Environment(df_train,reward=args.reward_type,state_space=1+len(list_indicators),tech_indicators=list_indicators,ws=window_size,start_point=24,end_point=len(df_train)-1,random=False,device=args.device,env_type="test")
    env_test=Environment(df_test,reward=args.reward_type,state_space=1+len(list_indicators),tech_indicators=list_indicators,ws=window_size,start_point=24,end_point=len(df_test)-1,random=False,device=args.device,env_type="test")
env = env_train

if args.algo == "vpg":
    if args.feature_extractor == "mlp":
        model = VPG_MLP(in_dim=env.observation_space.shape[0]*env.observation_space.shape[1],hidden_dim=120,out_dim=env.action_space.n).to(args.device)
    elif args.feature_extractor == "lstm":
        model = VPG_LSTM(in_dim=env.observation_space.shape[0],hidden_dim=120,num_layers=2,out_dim=env.action_space.n).to(args.device)
    elif args.feature_extractor == "transformer":
        model = VPG_TRANSFORMER(in_dim=env.observation_space.shape[0],hidden_dim=120,num_layers=2,out_dim=env.action_space.n).to(args.device)
elif args.algo == "dqn" or args.algo ==  "ddqn":
    if args.feature_extractor == "mlp":
        model = DQN_MLP(in_dim=env.observation_space.shape[0]*env.observation_space.shape[1],hidden_dim=120,out_dim=env.action_space.n).to(args.device)
    elif args.feature_extractor == "lstm":
        model = DQN_LSTM(in_dim=env.observation_space.shape[0],hidden_dim=120,out_dim=env.action_space.n,num_layers=2).to(args.device)
elif args.algo == "a2c":
    if args.feature_extractor == "mlp":
        model = A2C_MLP(in_dim=env.observation_space.shape[0]*env.observation_space.shape[1],hidden_dim=120,out_dim=env.action_space.n).to(args.device)
    elif args.feature_extractor == "lstm":
        model = A2C_LSTM(in_dim=env.observation_space.shape[0],hidden_dim=120,num_layers=2,out_dim=env.action_space.n).to(args.device)
    elif args.feature_extractor == "transformer":
        model = A2C_TRANSFORMER(in_dim=env.observation_space.shape[0],hidden_dim=120,num_layers=2,out_dim=env.action_space.n).to(args.device)
elif args.algo == "ppo":
    if args.feature_extractor == "mlp":
        pass
    elif args.feature_extractor == "lstm":
        model = PPO_LSTM(env.observation_space.shape[0],120,env.action_space.n,num_layers=2).to(args.device)
    elif args.feature_extractor == "transformer":
        model = PPO_TRANSFORMER(env.observation_space.shape[0],120,env.action_space.n,num_layers=2).to(args.device)

def test(model):
    wandb.init(project="DRL_CRYTO3_TEST",name=args.algo+"_"+args.feature_extractor+"_"+args.reward_type,entity="g0kul6")
    actions = []
    states = []
    rewards = []
    cumulative_reward = []
    model_path = path_dir + folder_name + "{}_{}_{}.pth".format(args.algo,args.feature_extractor,args.reward_type)
    model.load_state_dict(torch.load(model_path,map_location=args.device))
    env.reset()
    state = env.get_state()
    score = 0
    done = False
    step = 0
    while not done:
        if args.algo == "vpg":
            if args.feature_extractor == "mlp":
                action,log_prob,entropy = model.forward(torch.flatten(state.to(args.device)))
            elif args.feature_extractor == "lstm" or args.feature_extractor == "transformer":
                action,log_prob,entropy = model.forward(state.to(args.device),batch_size=1)
        elif args.algo == "dqn" or args.algo == "ddqn":
            if args.feature_extractor == "mlp":
                logits = model.forward(torch.flatten(state)).squeeze()
                action = torch.argmax(logits,dim=0)
            elif args.feature_extractor == "lstm":
                logits = model.forward(state,batch_size=1).squeeze()
                action = torch.argmax(logits,dim=0)
        elif args.algo == "a2c":
            if args.feature_extractor == "mlp":
                action,log_prob,value = model.forward(torch.flatten(state.to(args.device)))
            elif args.feature_extractor == "lstm" or args.feature_extractor == "transformer":
               action,log_prob,value = model.forward(state.to(args.device),batch_size=1)
        elif args.algo == "ppo":
            if args.feature_extractor == "mlp":
                pass
            elif args.feature_extractor == "lstm" or args.feature_extractor == "transformer":
                memory = Memory_PPO()
                action = model.act(state.to(args.device),memory)
                action = torch.tensor(action).to(args.device)
        reward,done,next_state = env.step(action)
        actions.append(action.cpu().item())
        rewards.append(reward.cpu().item())
        states.append(state.cpu().numpy())
        cumulative_reward.append(score)
        step = step +1
        score = score + reward.cpu().item()
        print("STEP",step,"CUMULATIVE_SCORE",score,"ACTION",action.cpu().item(),"REWARD",reward.cpu().item())
        wandb.log({"CUMULATIVE_REWARD":score,"IMEDIATE_REWARD":reward,"ACTION":action})
        state = next_state  
    np.save(test_dir+"action_{}_{}_{}.npy".format(args.algo,args.feature_extractor,args.reward_type),np.array(actions))
    np.save(test_dir+"state_{}_{}_{}.npy".format(args.algo,args.feature_extractor,args.reward_type),np.array(states))
    np.save(test_dir+"rewards_{}_{}_{}.npy".format(args.algo,args.feature_extractor,args.reward_type),np.array(rewards))
    np.save(test_dir+"cumulative_rewards_{}_{}_{}.npy".format(args.algo,args.feature_extractor,args.reward_type),np.array(cumulative_reward))
    

if __name__== '__main__':
    test(model=model)