
import os
import sys
from datetime import date
now = date.today()
path_dir = os.path.abspath(os.getcwd())
folder_name = "checkpoint{}/".format(now)
sys.path.append(path_dir+"/project_code")
path_checkpoint = path_dir + "/CHECKPOINT/" + folder_name
if not os.path.exists(path_checkpoint):
    os.makedirs(path_checkpoint)
    os.makedirs(path_checkpoint+"test/")
    os.makedirs(path_checkpoint+"point_images/")


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from drl_algo.memory import Memory_A2C,Memory_VPG,Memory_DQN,Memory_PPO
from drl_algo.models import A2C_LSTM,A2C_MLP,DQN_LSTM,DQN_MLP,VPG_LSTM,VPG_MLP,VPG_TRANSFORMER,A2C_TRANSFORMER
from drl_algo.ppo_train_util import PPO_Lstm, PPO_Transformer
import wandb
import random

def train_a2c_mlp(env,lr,gamma,epoch,device,reward_type):
    wandb.init(project="DRL_CRYPTO3_TRAIN",name="a2c_mlp_{}".format(reward_type),entity="g0kul6")
    a2c = A2C_MLP(in_dim=env.observation_space.shape[0]*env.observation_space.shape[1],hidden_dim=120,out_dim=env.action_space.n).to(device)
    optimizer = optim.Adam(a2c.parameters(),lr=lr)
    for i in range(epoch):
        env.reset()
        state = env.get_state()
        done = False
        memory = Memory_A2C(capacity=240)
        step = 0
        score = 0
        while not done:
            action,log_prob,value = a2c.forward(torch.flatten(state.to(device)))
            reward,done,next_state = env.step(action)
            score = score + reward.cpu().item()
            step = step + 1
            memory.push(value=value,action_log_prob=log_prob,reward=reward,done=done)
            state = next_state
        #train loop
        trajectory = memory.sample()
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(trajectory.reward), reversed(trajectory.done)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.stack(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        loss = 0
        actor_loss_log = 0
        value_loss_log = 0
        for value,reward,log_prob in zip(trajectory.value,rewards,trajectory.action_log_prob):
            advantage = reward - value.item()
            action_loss = -log_prob * advantage
            value_loss = F.smooth_l1_loss(value,reward)
            loss = loss + (action_loss+value_loss)
            actor_loss_log = actor_loss_log + action_loss
            value_loss_log = value_loss_log + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"loss":loss,"score":score,"actor_loss":actor_loss_log,"critic_loss":value_loss_log})
        print("Episode:",i,"loss:",loss.cpu().detach().item(),"score:",score,"actor_loss:",actor_loss_log.cpu().detach().item(),"critic_loss:",value_loss_log.cpu().detach().item())            
        torch.save(a2c.state_dict(),"{}a2c_mlp_{}.pth".format(path_checkpoint,reward_type))
                    


def train_a2c_lstm(env,lr,gamma,epoch,device,reward_type):
    wandb.init(project="DRL_CRYPTO3_TRAIN",name="a2c_lstm_{}".format(reward_type),entity="g0kul6")
    a2c = A2C_LSTM(in_dim=env.observation_space.shape[0],hidden_dim=120,num_layers=2,out_dim=env.action_space.n).to(device)
    optimizer = optim.Adam(a2c.parameters(),lr=lr)
    memory = Memory_A2C(capacity=240)
    for i in range(epoch):
        env.reset()
        state = env.get_state()
        done = False
        step = 0
        score = 0
        actions = []
        while not done:
            action,log_prob,value = a2c.forward(state.to(device),batch_size=1)
            reward,done,next_state = env.step(action)
            score = score + reward.cpu().item()
            step = step + 1
            memory.push(value=value,action_log_prob=log_prob,reward=reward,done=done)
            state = next_state
        #train loop
        trajectory = memory.sample()
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(trajectory.reward), reversed(trajectory.done)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.stack(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        loss = 0
        actor_loss_log = 0
        value_loss_log = 0
        for value,reward,log_prob in zip(trajectory.value,rewards,trajectory.action_log_prob):
            advantage = reward - value.item()
            action_loss = -log_prob * advantage
            value_loss = F.smooth_l1_loss(value,reward)
            loss = loss + (action_loss+value_loss)
            actor_loss_log = actor_loss_log + action_loss
            value_loss_log = value_loss_log + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        memory.clear()
        wandb.log({"loss":loss,"score":score,"actor_loss":actor_loss_log,"critic_loss":value_loss_log})
        torch.save(a2c.state_dict(),"{}a2c_lstm_{}.pth".format(path_checkpoint,reward_type))
        print("Episode",i,"loss:",loss.cpu().detach().item(),"score:",score,"actor_loss:",actor_loss_log.cpu().detach().item(),"critic_loss:",value_loss_log.cpu().detach().item())

def train_a2c_transformer(env,lr,gamma,epoch,device,reward_type):
    wandb.init(project="DRL_CRYPTO3_TRAIN",name="a2c_transformer_{}".format(reward_type),entity="g0kul6")
    a2c = A2C_TRANSFORMER(in_dim=env.observation_space.shape[0],hidden_dim=120,num_layers=2,out_dim=env.action_space.n).to(device)
    optimizer = optim.Adam(a2c.parameters(),lr=lr)
    memory = Memory_A2C(capacity=240)
    for i in range(epoch):
        env.reset()
        state = env.get_state()
        done = False
        step = 0
        score = 0
        actions = []
        while not done:
            action,log_prob,value = a2c.forward(state.to(device),batch_size=1)
            reward,done,next_state = env.step(action)
            score = score + reward.cpu().item()
            step = step + 1
            memory.push(value=value,action_log_prob=log_prob,reward=reward,done=done)
            state = next_state
        #train loop
        trajectory = memory.sample()
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(trajectory.reward), reversed(trajectory.done)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.stack(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        loss = 0
        actor_loss_log = 0
        value_loss_log = 0
        for value,reward,log_prob in zip(trajectory.value,rewards,trajectory.action_log_prob):
            advantage = reward - value.item()
            action_loss = -log_prob * advantage
            value_loss = F.smooth_l1_loss(value,reward)
            loss = loss + (action_loss+value_loss)
            actor_loss_log = actor_loss_log + action_loss
            value_loss_log = value_loss_log + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        memory.clear()
        wandb.log({"loss":loss,"score":score,"actor_loss":actor_loss_log,"critic_loss":value_loss_log})
        torch.save(a2c.state_dict(),"{}a2c_transformer_{}.pth".format(path_checkpoint,reward_type))
        print("Episode",i,"loss:",loss.cpu().detach().item(),"score:",score,"actor_loss:",actor_loss_log.cpu().detach().item(),"critic_loss:",value_loss_log.cpu().detach().item())                   

def train_vpg_mlp(env,lr,gamma,epoch,device,reward_type):
    wandb.init(project="DRL_CRYPTO3_TRAIN",name="vpg_mlp_{}".format(reward_type),entity="g0kul6")
    vpg = VPG_MLP(in_dim=env.observation_space.shape[0]*env.observation_space.shape[1],hidden_dim=120,out_dim=env.action_space.n).to(device)
    optimizer = optim.Adam(vpg.parameters(),lr=lr)
    for i in range(epoch):
        env.reset()
        state = env.get_state()
        done = False
        memory = Memory_VPG(capacity=240)
        step = 0
        score = 0
        while not done:
            action,log_prob,entropy = vpg.forward(torch.flatten(state.to(device)))
            reward,done,next_state = env.step(action)
            step = step + 1
            score = score + reward.cpu().item()
            memory.push(action_log_prob=log_prob,reward=reward,done=done,entropy=entropy)
            state = next_state
        #train loop
        trajectory = memory.sample()
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(trajectory.reward), reversed(trajectory.done)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.stack(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        loss = 0
        for reward,log_prob,entropy in zip(rewards,trajectory.action_log_prob,trajectory.entropy):
            weight = reward 
            action_loss = -log_prob * weight - 0.01 * entropy
            loss = loss + action_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        memory.clear()
        wandb.log({"loss":loss,"score":score})
        print("Episode:",i,"loss:",loss.cpu().detach().item(),"score:",score)            
        torch.save(vpg.state_dict(),"{}vpg_mlp_{}.pth".format(path_checkpoint,reward_type))


def train_vpg_lstm(env,lr,gamma,epoch,device,reward_type):
    wandb.init(project="DRL_CRYPTO3_TRAIN",name="vpg_lstm_{}".format(reward_type),entity="g0kul6")
    vpg = VPG_LSTM(in_dim=env.observation_space.shape[0],hidden_dim=120,num_layers=2,out_dim=env.action_space.n).to(device)
    optimizer = optim.Adam(vpg.parameters(),lr=lr)
    for i in range(epoch):
        env.reset()
        state = env.get_state()
        done = False
        memory = Memory_VPG(capacity=240)
        step = 0
        score = 0
        while not done:
            action,log_prob,entropy = vpg.forward(state.to(device),batch_size=1)
            reward,done,next_state = env.step(action)
            step = step + 1
            score = score + reward.cpu().item()
            memory.push(action_log_prob=log_prob,reward=reward,done=done,entropy=entropy)
            state = next_state
        #train loop
        trajectory = memory.sample()
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(trajectory.reward), reversed(trajectory.done)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + gamma * discounted_reward 
            rewards.insert(0, discounted_reward)
        rewards = torch.stack(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5) 
        loss = 0
        for reward,log_prob,entropy in zip(rewards,trajectory.action_log_prob,trajectory.entropy):
            weight = reward
            action_loss = -log_prob * weight #- 0.01 * entropy
            loss = loss + action_loss
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(vpg.parameters(),max_norm=2.0,norm_type=2)
        optimizer.step()
        memory.clear()
        wandb.log({"loss":loss,"score":score})
        torch.save(vpg.state_dict(),"{}vpg_lstm_{}.pth".format(path_checkpoint,reward_type))
        print("Episode",i,"loss:",loss.cpu().detach().item(),"score:",score)

def train_vpg_transformer(env,lr,gamma,epoch,device,reward_type):
    wandb.init(project="DRL_CRYPTO3_TRAIN",name="vpg_transformer_{}".format(reward_type),entity="g0kul6")
    vpg = VPG_TRANSFORMER(in_dim=env.observation_space.shape[0],hidden_dim=120,num_layers=2,out_dim=env.action_space.n).to(device)
    optimizer = optim.Adam(vpg.parameters(),lr=lr)
    for i in range(epoch):
        env.reset()
        state = env.get_state()
        done = False
        memory = Memory_VPG(capacity=240)
        step = 0
        score = 0
        while not done:
            action,log_prob,entropy = vpg.forward(state.to(device),batch_size=1)
            reward,done,next_state = env.step(action)
            step = step + 1
            score = score + reward.cpu().item()
            memory.push(action_log_prob=log_prob,reward=reward,done=done,entropy=entropy)
            state = next_state
        #train loop
        trajectory = memory.sample()
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(trajectory.reward), reversed(trajectory.done)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + gamma * discounted_reward 
            rewards.insert(0, discounted_reward)
        rewards = torch.stack(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5) 
        loss = 0
        for reward,log_prob,entropy in zip(rewards,trajectory.action_log_prob,trajectory.entropy):
            weight = reward
            action_loss = -log_prob * weight #- 0.01 * entropy
            loss = loss + action_loss
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(vpg.parameters(),max_norm=2.0,norm_type=2)
        optimizer.step()
        memory.clear()
        wandb.log({"loss":loss,"score":score})
        torch.save(vpg.state_dict(),"{}vpg_transformer_{}.pth".format(path_checkpoint,reward_type))
        print("Episode",i,"loss:",loss.cpu().detach().item(),"score:",score)


def train_dqn_lstm(env,lr,total_timesteps,target_network_frequency,batch_size,gamma,epsilon_decay,device,reward_type,max_epsilon=1.0,min_epsilon=0.1):
    wandb.init(project="DRL_CRYPTO3_TRAIN",name="dqn_lstm_{}".format(reward_type),entity="g0kul6")
    q_network = DQN_LSTM(in_dim=env.observation_space.shape[0],hidden_dim=120,out_dim=env.action_space.n,num_layers=2).to(device)
    optimizer = optim.Adam(q_network.parameters(),lr=lr)
    target_network = DQN_LSTM(in_dim=env.observation_space.shape[0],hidden_dim=120,out_dim=env.action_space.n,num_layers=2).to(device)
    target_network.load_state_dict(q_network.state_dict())
    rb = Memory_DQN(capacity=1000)
    epsilon = max_epsilon
    loss = 0
    for e in range(total_timesteps):
        env.reset()
        state = env.get_state()
        score = 0
        step = 0
        dones = False
        while not dones:
            step = step + 1
            if random.random() < epsilon:
                actions = torch.tensor(env.action_space.sample()).to(device)
            else:
                logits = q_network.forward(state,batch_size=1).squeeze()
                actions = torch.argmax(logits,dim=0)
            
            rewards,dones,next_state = env.step(actions)
            score = score + rewards
            dones = torch.tensor(dones).to(device)
            rb.push(state,next_state,actions,rewards,dones)
            state = next_state
            if len(rb)>batch_size:
                data = rb.sample(batch_size)
                epsilon = max(min_epsilon,epsilon-(max_epsilon - min_epsilon) * epsilon_decay)
                with torch.no_grad():
                    input = torch.stack(list(data.next_state))
                    # input = torch.reshape(input,(input.shape[0],input.shape[1]*input.shape[2]))
                    target_max,_ = target_network.forward(input,batch_size=batch_size).max(dim=1)
                    td_target = torch.stack(list(data.reward)).unsqueeze(dim=1) + gamma * target_max * (~torch.stack(list(data.done))).unsqueeze(dim=1)
                input1 = torch.stack(list(data.state))
                # input1 = torch.reshape(input1,(input1.shape[0],input1.shape[1]*input1.shape[2]))
                q_value = q_network.forward(input1,batch_size=batch_size).squeeze()
                q_value = q_value.gather(1,torch.stack(list(data.action)).unsqueeze(dim=1))
                loss = F.mse_loss(td_target,q_value)
                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if e%target_network_frequency == 0:
                    target_network.load_state_dict(q_network.state_dict())
        wandb.log({"loss":loss,"score":score,"epsioln":epsilon})
        print("Episode",e,"loss:",loss.cpu().detach().item(),"score:",score.cpu().item(),"epsilom:",epsilon)
        torch.save(q_network.state_dict(),"{}dqn_lstm_{}.pth".format(path_checkpoint,reward_type))

def train_dqn_mlp(env,lr,total_timesteps,target_network_frequency,batch_size,gamma,epsilon_decay,device,reward_type,max_epsilon=1.00,min_epsilon=0.1):
    wandb.init(project="DRL_CRYPTO3_TRAIN",name="dqn_mlp_{}".format(reward_type),entity="g0kul6")
    q_network = DQN_MLP(in_dim=env.observation_space.shape[0]*env.observation_space.shape[1],hidden_dim=120,out_dim=env.action_space.n).to(device)
    optimizer = optim.Adam(q_network.parameters(),lr=lr)
    target_network = DQN_MLP(in_dim=env.observation_space.shape[0]*env.observation_space.shape[1],hidden_dim=120,out_dim=env.action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())
    rb = Memory_DQN(capacity=1000)
    epsilon = max_epsilon
    loss = 0
    for e in range(total_timesteps):
        env.reset()
        state = env.get_state()
        score = 0
        step = 0
        dones = False
        while not dones:
            step = step + 1
            if random.random() < epsilon:
                actions = torch.tensor(env.action_space.sample()).to(device)
            else:
                logits = q_network.forward(torch.flatten(state)).squeeze()
                actions = torch.argmax(logits,dim=0)
            
            rewards,dones,next_state = env.step(actions)
            score = score + rewards
            dones = torch.tensor(dones).to(device)
            rb.push(state,next_state,actions,rewards,dones)
            state = next_state
            if len(rb)>batch_size:
                data = rb.sample(batch_size)
                epsilon = max(min_epsilon,epsilon-(max_epsilon - min_epsilon) * epsilon_decay)
                with torch.no_grad():
                    input = torch.stack(list(data.next_state))
                    input = torch.reshape(input,(input.shape[0],input.shape[1]*input.shape[2]))
                    target_max,_ = target_network.forward(input).max(dim=1)
                    td_target = torch.stack(list(data.reward)).unsqueeze(dim=1) + gamma * target_max * (~torch.stack(list(data.done))).unsqueeze(dim=1)
                input1 = torch.stack(list(data.state))
                input1 = torch.reshape(input1,(input1.shape[0],input1.shape[1]*input1.shape[2]))
                q_value = q_network.forward(input1).squeeze()
                q_value = q_value.gather(1,torch.stack(list(data.action)).unsqueeze(dim=1))
                loss = F.mse_loss(td_target,q_value)
                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if e%target_network_frequency == 0:
                    target_network.load_state_dict(q_network.state_dict())
        wandb.log({"loss":loss,"score":score,"epsioln":epsilon})
        print("Episode",e,"loss:",loss.cpu().detach().item(),"score:",score.cpu().item(),"epsilom:",epsilon)
        torch.save(q_network.state_dict(),"{}dqn_mlp_{}.pth".format(path_checkpoint,reward_type))

def train_ddqn_lstm(env,lr,total_timesteps,target_network_frequency,batch_size,gamma,epsilon_decay,device,reward_type,max_epsilon=1.0,min_epsilon=0.1):
    wandb.init(project="DRL_CRYPTO3_TRAIN",name="ddqn_lstm_{}".format(reward_type),entity="g0kul6")
    q_network = DQN_LSTM(in_dim=env.observation_space.shape[0],hidden_dim=120,out_dim=env.action_space.n,num_layers=2).to(device)
    optimizer = optim.Adam(q_network.parameters(),lr=lr)
    target_network = DQN_LSTM(in_dim=env.observation_space.shape[0],hidden_dim=120,out_dim=env.action_space.n,num_layers=2).to(device)
    target_network.load_state_dict(q_network.state_dict())
    rb = Memory_DQN(capacity=1000)
    epsilon = max_epsilon
    loss = 0
    for e in range(total_timesteps):
        env.reset()
        state = env.get_state()
        score = 0
        step = 0
        dones = False
        while not dones:
            step = step + 1
            if random.random() < epsilon:
                actions = torch.tensor(env.action_space.sample()).to(device)
            else:
                logits = q_network.forward(state,batch_size=1).squeeze()
                actions = torch.argmax(logits,dim=0)
            
            rewards,dones,next_state = env.step(actions)
            score = score + rewards
            dones = torch.tensor(dones).to(device)
            rb.push(state,next_state,actions,rewards,dones)
            state = next_state
            if len(rb)>batch_size:
                data = rb.sample(batch_size)
                epsilon = max(min_epsilon,epsilon-(max_epsilon - min_epsilon) * epsilon_decay)
                input1 = torch.stack(list(data.state))
                # input1 = torch.reshape(input1,(input1.shape[0],input1.shape[1]*input1.shape[2]))
                q_value = q_network.forward(input1,batch_size=batch_size).squeeze()
                with torch.no_grad():
                    input = torch.stack(list(data.next_state))
                    # input = torch.reshape(input,(input.shape[0],input.shape[1]*input.shape[2]))
                    max_actions = []
                    for i in q_value:
                        max_actions.append(torch.argmax(i,dim=0))
                    max_actions = torch.tensor(max_actions).to(device)
                    target = target_network.forward(input,batch_size=batch_size)
                    td_target = torch.stack(list(data.reward)).unsqueeze(dim=1) + gamma * target.gather(1,max_actions.unsqueeze(dim=1)) * (~torch.stack(list(data.done))).unsqueeze(dim=1)
                q_value = q_value.gather(1,torch.stack(list(data.action)).unsqueeze(dim=1))
                loss = F.mse_loss(td_target,q_value)
                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if e%target_network_frequency == 0:
                    target_network.load_state_dict(q_network.state_dict())
        wandb.log({"loss":loss,"score":score,"epsioln":epsilon})
        print("Episode",e,"loss:",loss.cpu().detach().item(),"score:",score.cpu().item(),"epsilom:",epsilon)
        torch.save(q_network.state_dict(),"{}ddqn_lstm_{}.pth".format(path_checkpoint,reward_type))

def train_ppo_lstm(env, state_dim, action_dim, max_episodes, update_timestep, K_epochs, eps_clip, gamma, lr, betas,reward_type,device):
    wandb.init(project="DRL_CRYPTO3_TRAIN",name="ppo_lstm_{}".format(reward_type),entity="g0kul6")
    memory = Memory_PPO()
    ppo = PPO_Lstm(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip,reward_type=reward_type,device=device)
    running_reward, avg_length, time_step = 0, 0, 0
    # training loop
    for i_episode in range(1, max_episodes+1):
        env.reset()
        state = env.get_state()
        step = 0
        done = False
        actions = []
        running_reward = 0
        while not done:
            time_step += 1
            # Run old policy
            action = ppo.select_action(state, memory)
            actions.append(action)
            reward,done,next_state = env.step(action)
            step = step + 1
            running_reward+= reward
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            state = next_state
            if time_step % update_timestep == 0:
                loss,l_r = ppo.update(memory)
                memory.clear_memory()
                time_step = 0
        print('Episode',i_episode,'reward',running_reward.cpu().item(),'loss',loss.cpu().item())
        wandb.log({"score":running_reward,"loss":loss})

def train_ppo_transformer(env, state_dim, action_dim, max_episodes, update_timestep, K_epochs, eps_clip, gamma, lr, betas,reward_type,device):
    wandb.init(project="DRL_CRYPTO3_TRAIN",name="ppo_transformer_{}".format(reward_type),entity="g0kul6")
    memory = Memory_PPO()
    ppo = PPO_Transformer(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip,reward_type=reward_type,device=device)
    running_reward, avg_length, time_step = 0, 0, 0
    # training loop
    for i_episode in range(1, max_episodes+1):
        env.reset()
        state = env.get_state()
        step = 0
        done = False
        actions = []
        running_reward = 0
        while not done:
            time_step += 1
            # Run old policy
            action = ppo.select_action(state, memory)
            actions.append(action)
            reward,done,next_state = env.step(action)
            step = step + 1
            running_reward+= reward
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            state = next_state
            if time_step % update_timestep == 0:
                loss,l_r = ppo.update(memory)
                memory.clear_memory()
                time_step = 0
        print('Episode',i_episode,'reward',running_reward.cpu().item(),'loss',loss.cpu().item())
        wandb.log({"score":running_reward,"loss":loss})


