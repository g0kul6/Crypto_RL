#replay buffer
#memory for polocy based algos 

from collections import namedtuple
import torch
import numpy as np
import random
from collections import namedtuple, deque

Transition_DQN = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'done'))

class Memory_DQN(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, done):
        self.memory.append(Transition_DQN(state, next_state, action, reward, done))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition_DQN(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)

Transition_A2C = namedtuple('Transition', ('value', 'action_log_prob', 'reward', 'done'))

class Memory_A2C(object):
    def __init__(self,capacity):
        # self.memory = deque(maxlen=capacity)
        self.memory = []

    def push(self, value, action_log_prob, reward, done):
        self.memory.append(Transition_A2C(value, action_log_prob, reward, done))

    def sample(self):
        memory = self.memory
        return Transition_A2C(*zip(*memory))

    def clear(self):
        self.memory = []
        
    def __len__(self):
        return len(self.memory)



Transition_VPG = namedtuple('Transition', ('action_log_prob', 'reward', 'done','entropy'))

class Memory_VPG(object):
    def __init__(self,capacity):
        # self.memory = deque(maxlen=capacity)
        self.memory = []

    def push(self, action_log_prob, reward, done,entropy):
        self.memory.append(Transition_VPG(action_log_prob, reward, done,entropy))

    def sample(self):
        memory = self.memory
        return Transition_VPG(*zip(*memory))
    def clear(self):
        self.memory = []
    def __len__(self):
        return len(self.memory)

class Memory_PPO:   # collected from old policy
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.logprobs = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]