from math import gamma
import numpy as np

"""
This file contain all the hyperparmas
Look for available indicators in data.txt
"""


list_indicators=['trend_macd','momentum_rsi','trend_cci','trend_adx']

window_size = 24
start_point = 24 #should be atleast 12 

actor_lr = 1e-4
critic_lr = 1e-4

max_episodes=10000

gamma = 0.99

