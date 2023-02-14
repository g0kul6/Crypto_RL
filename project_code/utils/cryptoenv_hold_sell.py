import torch
import numpy as np
import gym
from gym import  spaces
import numpy as np


class Environment(gym.Env):
    """Definition of the trading environment .
    Attributes:
        data (pandas.DataFrame): Time serie to be considered within the environment.
        t (:obj:`int`): Current time instant we are considering.
        profits (:obj:`float`): profit of the agent at time self.t
        agent_positions(:obj:`list` :obj:`float`): list of the positions
           currently owned by the agent.
        agent_position_value(:obj:`float`): current value of open positions
           (positions in self.agent_positions)
    """

    def __init__(self, data, reward,state_space,ws,tech_indicators,start_point,end_point,device,random=True,env_type="train"):
        """
        Creates the environment. Note: Before using the environment you must call
        the Environment.reset() method.
        Args:
           data (:obj:`pd.DataFrane`): Time serie to be initialize the environment.
           reward (:obj:`str`): Type of reward function to use, either sharpe ratio
              "sr" or profit function "profit"
        """
        self.env_type = env_type
        self.data = data
        self.ws=ws
        self.start_point=start_point
        self.end_point=end_point
        self.random=random
        self.tech_indicators=tech_indicators
        self.state_space=state_space
        self.reward_f = reward if reward == "sr" else "profit"
        self.action_space=spaces.Discrete(2)
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(self.state_space,self.ws))
        self.device = device

    def reset(self):
        """
        Reset the environment or makes a further step of initialization if called
        on an environment never used before. It must always be called before .step()
        method to avoid errors.
        """
        self.epsiode_reward = 0
        self.episode_actions = []
        if self.random:
            self.t = np.random.randint(self.ws,len(self.data)-240)
        else :
            self.t = self.start_point
        self.done = False
        self.rewards = []
        self.profits = [0 for e in range(len(self.data))]
        self.sharpe_ratios = [0 for e in range(len(self.data))]
        self.agent_positions = []
        self.positions = []
        self.steps = 0
        self.no_buys = 0
        self.no_sells = 0
        self.no_holds = 0
    def get_state(self):
        """
            Return the current state of the environment. NOTE: if called after
            Environment.step() it will return the next state.
        """
        if not self.done:
            next_state = [self.data.iloc[self.t - self.ws:self.t, :]['Close']]
            for i in self.tech_indicators:
                next_state.append(self.data.iloc[self.t-self.ws:self.t, :][i])
            return torch.tensor(np.array(next_state), device=self.device, dtype=torch.float)

        else:
            return None

    def step(self, act):
        """
        Perform the action of the Agent on the environment, computes the reward
        and update some datastructures to keep track of some econometric indexes
        during time.
        Args:
           act (:obj:`int`): Action to be performed on the environment.
        Returns:
            reward (:obj:`torch.tensor` :dtype:`torch.float`): the reward of
                performing the action on the current env state.
            self.done (:obj:`bool`): A boolean flag telling if we are in a final
                state
            current_state (:obj:`torch.tensor` :dtype:`torch.float`):
                the state of the environment after the action execution.
        """
        self.episode_actions.append(act)
        reward = 0
        # EXECUTE THE ACTION (act = 0: stay, 1: sell)
        if act == 0:  # Do Nothing
            self.no_holds = self.no_holds + 1
            pass
        if act == 1:  # Sell
            self.no_buys = 0
            self.no_holds = 0
            self.no_sells = self.no_sells + 1
            profit = self.data.iloc[self.t]["Close"] - self.data.iloc[self.t-1]["Close"]
            self.profits[self.t] = profit
        self.steps += 1
        self.epsiode_reward = self.epsiode_reward + reward

        # REWARD
        if self.env_type == "train":
            if act == 0:
                # if sell
                if_sell_profit = self.data.iloc[self.t]["Close"] - self.data.iloc[self.t-1]["Close"]
                if if_sell_profit<0:
                    reward = 1
                elif if_sell_profit == 0:
                    reward = 1
                elif if_sell_profit>0:
                    reward = -1
                # if self.data.iloc[self.t]["Unorm_rsi"]>70:
                #     reward = -1 + reward
                # elif self.data.iloc[self.t]["Unorm_rsi"]<30:
                #     reward = 1 + reward
            elif act == 1:
                p = self.profits[self.t]
                if p>0:
                    reward = 1 
                elif p == 0:
                    reward = 0
                elif p < 0 :
                    reward = -1
                # if self.data.iloc[self.t]["Unorm_rsi"]>70:
                #     reward = 1 + reward
                # elif self.data.iloc[self.t]["Unorm_rsi"]<30:
                #     reward = -1 + reward
            if self.random:
                if (self.steps == 200):
                    self.done = True
            else:
                if (self.t == self.end_point):
                    self.done = True
        elif self.env_type == "test":
            reward = self.profits[self.t]
        self.steps += 1
        # Give Terminal state 
        if self.random:
            if (self.steps == 240):
                self.done = True
        else:
            if (self.t == self.end_point):
                self.done = True
        # Update t
        if self.env_type == "train":
            self.t += 1
        elif self.env_type == "test":
            self.t += 1
        # GET NEXT STATE
        state = [self.data.iloc[self.t-self.ws:self.t, :]['Close']]
        for i in self.tech_indicators:
            state.append(self.data.iloc[self.t-self.ws:self.t, :][i])

        self.prev_action = act
        return torch.tensor([reward], device=self.device, dtype=torch.float), self.done, torch.tensor(np.array(state), device=self.device, dtype=torch.float) # reward, done, current_state