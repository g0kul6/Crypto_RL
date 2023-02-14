import torch
import numpy as np
import gym
from gym import  spaces
import numpy as np



class Environment(gym.Env):
    """
    Definition of the trading environment .
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
        self.action_space=spaces.Discrete(3)
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(self.state_space,self.ws))
        self.device = device

    def reset(self):
        """
        Reset the environment or makes a further step of initialization if called
        on an environment never used before. It must always be called before .step()
        method to avoid errors.
        """
        if self.random:
            self.t = np.random.randint(self.ws,len(self.data)-240)
        else :
            self.t = self.start_point
        self.done = False
        self.profits = [0 for e in range(len(self.data))]
        self.sharpe_ratios = [0 for e in range(len(self.data))]
        self.init_price = self.data.iloc[self.t-1, :]['Close']
        self.agent_positions = []
        self.steps = 0
        self.stocks = 0
        self.no_buys = 0
        self.no_sells = 0
        self.no_holds = 0
    def get_state(self):
        """
            Return the current state of the environment. NOTE: if called after
            Environment.step() it will return the next state.
        """
        if not self.done:
            next_state = [self.data.iloc[self.t - self.ws:self.t]['Close']]
            for i in self.tech_indicators:
                next_state.append(self.data.iloc[self.t-self.ws:self.t][i])
            return torch.tensor(np.array(next_state), device=self.device, dtype=torch.float)

        else:
            return None

    def step(self, act):
        """
        Perform the action of the Agent on the environment, computes the reward
        and update some datastructures to keep track of some econometric indexes
        during time.
        """
        reward = 0
        sell = True
    
        # EXECUTE THE ACTION (act = 0: stay, 1: buy, 2: sell)
        if act == 0:  # Do Nothing
            self.no_holds = self.no_holds + 1
            pass
        if act == 1:  # Buy
            self.no_buys = 0
            self.no_buys = self.no_buys + 1
            self.agent_positions.append(self.data.iloc[self.t]['Close'])
            self.stocks = self.stocks + 1
        if act == 2:  # Sell
            if len(self.agent_positions) == 0:
                sell = False
            if sell:
                self.no_buys = 0
                self.no_holds = 0
                self.no_sells = self.no_sells + 1
                profit_sum = 0
                for position in self.agent_positions:
                    profit_sum += (self.data.iloc[self.t]['Close'] - position) # profit = close - my_position for each my_position "
                self.profits[self.t] = profit_sum
                self.agent_positions = []
                self.stocks = 0

        if self.env_type == "train":
            if self.reward_f == "profit":
                r_p = 0
                r_m = 0
                r_if = 0
                if act == 2:
                    # profits based reward
                    # r_p = self.profits[self.t]
                    if self.profits[self.t] > 0:
                        r_p = 1
                    elif self.profits[self.t] == 0:
                        r_p = 0
                    elif self.profits[self.t] < 0:
                        r_p = -1
                    # # rsi momentum based
                    # if self.data.iloc[self.t]["Unorm_rsi"] > 80:
                    #     r_m = 1
                    # elif self.data.iloc[self.t]["Unorm_rsi"] < 20:
                    #     r_m = -1
                    reward = r_m + r_p + r_if
                    # wrong action
                    if not sell:
                        reward = -5
                        
                if act == 1:
                    # rsi momentum based
                    if self.data.iloc[self.t]["Unorm_rsi"] > 70:
                        r_m = -1
                    elif self.data.iloc[self.t]["Unorm_rsi"] < 30:
                        r_m = -1
                    else:
                        r_m = 1
                    #if  sell 
                    # if_sell_profit = 0
                    # for position in self.agent_positions:
                    #     if_sell_profit += (self.data.iloc[self.t]['Close'] - position) # profit = close - my_position for each my_position "
                    # if if_sell_profit > 0:
                    #     r_if = -1
                    # elif if_sell_profit <= 0:
                    #     r_if = 1
                    reward = r_m + r_if + r_p


                if act == 0:
                    # rsi momentum based
                    if self.data.iloc[self.t]["Unorm_rsi"] > 70:
                        r_m = -1
                    elif self.data.iloc[self.t]["Unorm_rsi"] < 30:
                        r_m = 1
                    #if  sell 
                    # if_sell_profit = 0
                    # for position in self.agent_positions:
                    #     if_sell_profit += (self.data.iloc[self.t]['Close'] - position) # profit = close - my_position for each my_position "
                    # if if_sell_profit > 0:
                    #     r_if = -1
                    # elif if_sell_profit <= 0:
                    #     r_if = 1

                    reward = r_m + r_p + r_if
                    
        if self.env_type == "test":
            if self.reward_f == "profit":
                reward = self.profits[self.t]
    
        # UPDATE THE STATE
        self.steps += 1
        if self.env_type == "train":
            self.t += 1
        elif self.env_type == "test":
            self.t += 1
        if self.random:
            if (self.steps == 240):
                self.done = True
        else:
            if (self.t == self.end_point):
                self.done = True
        # GET NEXT STATE
        state = [self.data.iloc[self.t-self.ws:self.t]['Close']]
        for i in self.tech_indicators:
            state.append(self.data.iloc[self.t-self.ws:self.t][i])
        self.prev_action = act
        return torch.tensor([reward], device=self.device, dtype=torch.float), self.done, torch.tensor(np.array(state), device=self.device, dtype=torch.float) # reward, done, current_state