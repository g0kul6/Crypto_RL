# 🤖 Crypto_RL 🤖

#### ALGO:

* *vpg* - *Vanilla Policy Gradient*
* *a2c* - *Advantage Actor Critic*
* ppo - *Proximal Policy Optimization*
* dqn - *Deep Q-Netowork*
* ddqn - *Dueling Deep Q-Netowork*

#### FEATURE_EXTRACTOR:

* *mlp*
* *lstm*
* *transformer*

#### REWARD_TYPE:

* *profit*
* *sr*  (!not yet added)

#### DEVICE:

* cuda

#### HYPERPARAMS:

* *lr* - *actor and critic lr*
* *gamma* - *discount factor*
* epochs

### **Before running code :**

```bash
conda activate rl
mkdir CHECKPOINT
mkdir dataset
```

### **Training :**

```bash
python project_code/drl_algo/train.py --algo ALGO --feature_extractor FEATURE_EXTRACTOR --reward_type REWARD_TYPE --lr LR --gamma GAMMA --device DEVICE --epochs EPOCH
```

### **Testing :**

```bash
python project_code/drl_algo/test.py --algo ALGO --feature_extractor FEATURE_EXTRACTOR --reward_type REWARD_TYPE --device DEVICE
```

### **Ploting Action Plots :**

```bash
#if buy_Sell
python project_code/utils/plot_buy_sell.py --algo ALGO --feature_extractor FEATURE_EXTRACTOR --reward_type REWARD_TYPE
#if hold_Sell
python project_code/utils/plot_hold_sell.py --algo ALGO --feature_extractor FEATURE_EXTRACTOR --reward_type REWARD_TYPE
#if hold_buy_sell
python project_code/utils/plot_hold_buy_sell.py --algo ALGO --feature_extractor FEATURE_EXTRACTOR --reward_type REWARD_TYPE
```

### **Ploting Close Values :**

MODE: train/test
PROJECT_NAME: wandb project name used for testing

```bash
python project_code/utils/plot_states.py --mode MODE --project_name PROJECT_NAME
```
