import os
import sys
import torch
from datetime import date
now = date.today()
path_dir = os.path.abspath(os.getcwd())
folder_name = "/checkpoint{}/".format(now)
sys.path.append(path_dir+"/project_code")
path_checkpoint = path_dir + "/CHECKPOINT/" + folder_name
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from drl_algo.models import PPO_LSTM,PPO_TRANSFORMER

class PPO_Lstm:
    def __init__(self, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, device,restore=False, ckpt=None,reward_type="profit"):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.reward_type = reward_type
        # current policy
        self.policy = PPO_LSTM(state_dim,120,action_dim,num_layers=2).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr,betas=betas)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
        # old policy: initialize old policy with current policy's parameter
        self.old_policy = PPO_LSTM(state_dim,120,action_dim,num_layers=2).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MSE_loss = nn.MSELoss()	# to calculate critic loss

    def select_action(self, state, memory):
        # state = state.reshape(1, -1)  # flatten the state
        return self.old_policy.act(state, memory)

    def update(self, memory):
        # Monte Carlo estimation of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(self.device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(self.device).detach()

        # Train policy for K epochs: sampling and updating
        for _ in range(self.K_epochs):
            # Evaluate old actions and values using current policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions,batch_size=old_states.shape[0])

            # Importance ratio: p/q
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Advantages
            advantages = rewards - state_values.detach()  # old states' rewards - old states' value( evaluated by current policy)

            # Actor loss using Surrogate loss
            surr1 = ratios * advantages 
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages 
            actor_loss = - torch.min(surr1, surr2)

            # Critic loss: critic loss - entropy
            critic_loss = 0.5 * self.MSE_loss(rewards, state_values) - 0.01 *  dist_entropy

            # Total loss
            loss = (actor_loss + critic_loss ) 
            loss = loss.mean()
            for g in self.optimizer.param_groups:
                a = g['lr']
            # Backward gradients
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # self.scheduler.step()
         # Copy new weights to old_policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        memory.clear_memory()
        torch.save(self.policy.state_dict(),"{}ppo_lstm_{}.pth".format(path_checkpoint,self.reward_type))
        return loss,a

class PPO_Transformer:
    def __init__(self, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, device,restore=False, ckpt=None,reward_type="profit"):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.reward_type = reward_type
        # current policy
        self.policy = PPO_TRANSFORMER(state_dim,120,action_dim,num_layers=2).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr,betas=betas)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
        # old policy: initialize old policy with current policy's parameter
        self.old_policy = PPO_TRANSFORMER(state_dim,120,action_dim,num_layers=2).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MSE_loss = nn.MSELoss()	# to calculate critic loss

    def select_action(self, state, memory):
        # state = state.reshape(1, -1)  # flatten the state
        return self.old_policy.act(state, memory)

    def update(self, memory):
        # Monte Carlo estimation of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(self.device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(self.device).detach()

        # Train policy for K epochs: sampling and updating
        for _ in range(self.K_epochs):
            # Evaluate old actions and values using current policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions,batch_size=old_states.shape[0])

            # Importance ratio: p/q
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Advantages
            advantages = rewards - state_values.detach()  # old states' rewards - old states' value( evaluated by current policy)

            # Actor loss using Surrogate loss
            surr1 = ratios * advantages 
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages 
            actor_loss = - torch.min(surr1, surr2)

            # Critic loss: critic loss - entropy
            critic_loss = 0.5 * self.MSE_loss(rewards, state_values) - 0.01 *  dist_entropy

            # Total loss
            loss = (actor_loss + critic_loss ) 
            loss = loss.mean()
            for g in self.optimizer.param_groups:
                a = g['lr']
            # Backward gradients
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # self.scheduler.step()
         # Copy new weights to old_policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        memory.clear_memory()
        torch.save(self.policy.state_dict(),"{}ppo_transformer_{}.pth".format(path_checkpoint,self.reward_type))
        return loss,a