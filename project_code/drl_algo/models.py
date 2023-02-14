import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
This file has pytorch models for different algorithms

Instruction for mlp:
in_dim = env.observation_space.shape[0]*env.observation.shape[1]
out_dim = env.action_spacae.n or 1 (depending on policy based and value based)

Instruction for lstm:
in_dim = env.observation_space.shape[0]
out_dim = env.action_spacae.n or 1 (depending on policy based and value based)
"""

#vpg mlp
class VPG_MLP(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim):
        super().__init__()
        #actor
        self.network_actor = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim),
            nn.Softmax()
        )

    
    def forward(self,x):
        action_prob = self.network_actor(x)
        action_distribution = torch.distributions.Categorical(action_prob)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
        return action,log_prob,entropy

#vpg lstm
class VPG_LSTM(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,num_layers):
        super().__init__()
        #lstm 
        self.network_lstm = nn.Sequential(
            nn.LSTM(in_dim,hidden_dim,num_layers,batch_first=True)
        )
        #actor
        self.network_actor = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim),
            nn.Softmax()
        )
    
    def forward(self,x,batch_size):
        if batch_size==1:
            x = torch.reshape(x,(batch_size,x.shape[1],x.shape[0]))
        else:    
            x = torch.reshape(x,(batch_size,x.shape[2],x.shape[1]))
        out,_ = self.network_lstm(x)
        action_prob = self.network_actor(out[:,-1,:])
        action_distribution = torch.distributions.Categorical(action_prob)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
        return action,log_prob,entropy

#vpg tranformer
class VPG_TRANSFORMER(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,num_layers):
        super().__init__()
        #tranformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=in_dim)
        self.network_transformer = nn.Sequential(
            nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        )
        #actor
        self.network_actor = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim),
            nn.Softmax()
        )
    
    def forward(self,x,batch_size):
        if batch_size==1:
            x = torch.reshape(x,(x.shape[1],batch_size,x.shape[0]))
        else:    
            x = torch.reshape(x,(x.shape[2],batch_size,x.shape[1]))
        out = self.network_transformer(x)
        action_prob = self.network_actor(out[-1,:,:])
        action_distribution = torch.distributions.Categorical(action_prob)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
        return action,log_prob,entropy

#a2c mlp
class A2C_MLP(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim):
        super().__init__()
        #feature extractor
        self.common_network = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU()
        )
        #actor
        self.network_actor = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim),
            nn.Softmax()
        )
        #critic
        self.network_critic = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    
    def forward(self,x):
        out = self.common_network(x)
        action_prob = self.network_actor(out)
        action_distribution = torch.distributions.Categorical(action_prob)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        value = self.network_critic(out)
        return action,log_prob,value

#a2c lstm
class A2C_LSTM(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,num_layers):
        super().__init__()
        #lstm 
        self.network_lstm = nn.Sequential(
            nn.LSTM(in_dim,hidden_dim,num_layers,batch_first=True)
        )
        #actor
        self.network_actor = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim),
            nn.Softmax()
        )
        #critic
        self.network_critic = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    
    def forward(self,x,batch_size):
        if batch_size==1:
            x = torch.reshape(x,(batch_size,x.shape[1],x.shape[0]))
        else:    
            x = torch.reshape(x,(batch_size,x.shape[2],x.shape[1]))
        out,_ = self.network_lstm(x)
        action_prob = self.network_actor(out[:,-1,:])
        action_distribution = torch.distributions.Categorical(action_prob)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        value = self.network_critic(out[:,-1,:])
        return action,log_prob,value

#a2c transformer
class A2C_TRANSFORMER(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,num_layers):
        super().__init__()
        #tranformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=in_dim)
        self.network_transformer = nn.Sequential(
            nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        )
        #actor
        self.network_actor = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim),
            nn.Softmax()
        )
        #critic
        self.network_critic = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    
    def forward(self,x,batch_size):
        if batch_size==1:
            x = torch.reshape(x,(x.shape[1],batch_size,x.shape[0]))
        else:    
            x = torch.reshape(x,(x.shape[2],batch_size,x.shape[1]))
        out = self.network_transformer(x)
        action_prob = self.network_actor(out[-1,:,:])
        action_distribution = torch.distributions.Categorical(action_prob)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        value = self.network_critic(out[-1,:,:])
        return action,log_prob,value

#dqn mlp
class DQN_MLP(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim)
        )
    def forward(self,x):
        return self.network(x)

#dqn lstm
class DQN_LSTM(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,num_layers):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim)
        )

    def forward(self,x,batch_size):
        if batch_size==1:
            x = torch.reshape(x,(batch_size,x.shape[1],x.shape[0]))
        else:    
            x = torch.reshape(x,(batch_size,x.shape[2],x.shape[1]))
        out,_ = self.lstm(x)
        q_value = self.fc(out[:,-1,:])
        return q_value 


class PPO_MLP(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim):
        super(PPO_MLP,self).__init__()
        # actor
        self.actor = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim),
            nn.Softmax(dim=1)
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def act(self,state,memory):
        logits = self.actor(state)
        distribution = torch.distributions.Categorical(logits)
        action = distribution.sample()
        action_logprob = distribution.log_prob(action)
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action.item()

    def evaluate(self,state,action):
        state_value = self.critic(state)
        
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)	
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO_LSTM(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,num_layers):
        super(PPO_LSTM,self).__init__()
        #lstm 
        self.network_lstm = nn.Sequential(
            nn.LSTM(in_dim,hidden_dim,num_layers,batch_first=True)
        )
        # actor
        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim),
            nn.Softmax(dim=1)
        )
        # critic
        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def act(self,state,memory,batch_size=1):
        memory.states.append(state)
        if batch_size==1:
            
            state = torch.reshape(state,(batch_size,state.shape[1],state.shape[0]))
        else:    
            state = torch.reshape(state,(batch_size,state.shape[2],state.shape[1]))
        state,_ = self.network_lstm(state)
        logits = self.actor(state[:,-1,:])
        distribution = torch.distributions.Categorical(logits)
        action = distribution.sample()
        action_logprob = distribution.log_prob(action)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action.item()

    def evaluate(self,state,action,batch_size=1):
        if batch_size==1:
            state = torch.reshape(state,(batch_size,state.shape[1],state.shape[0]))
        else:    
            state = torch.reshape(state,(batch_size,state.shape[2],state.shape[1]))
        state,_ = self.network_lstm(state)
        state_value = self.critic(state[:,-1,:])
        
        action_probs = self.actor(state[:,-1,:])
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)	
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO_TRANSFORMER(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,num_layers):
        super(PPO_TRANSFORMER,self).__init__()
        #tranformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=in_dim)
        self.network_transformer = nn.Sequential(
            nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        )
        # actor
        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim),
            nn.Softmax(dim=1)
        )
        # critic
        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def act(self,state,memory,batch_size=1):
        memory.states.append(state)
        if batch_size==1:
            
            state = torch.reshape(state,(state.shape[1],batch_size,state.shape[0]))
        else:    
            state = torch.reshape(state,(state.shape[2],batch_size,state.shape[1]))
        state = self.network_transformer(state)
        logits = self.actor(state[-1,:,:])
        distribution = torch.distributions.Categorical(logits)
        action = distribution.sample()
        action_logprob = distribution.log_prob(action)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action.item()

    def evaluate(self,state,action,batch_size=1):
        if batch_size==1:
            state = torch.reshape(state,(state.shape[1],batch_size,state.shape[0]))
        else:    
            state = torch.reshape(state,(state.shape[2],batch_size,state.shape[1]))
        state = self.network_transformer(state)
        state_value = self.critic(state[-1,:,:])
        
        action_probs = self.actor(state[-1,:,:])
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)	
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy
