import os
import sys
path_dir = os.path.abspath(os.getcwd())
sys.path.append(path_dir+"/project_code")
data_dir = path_dir + "/dataset/BITCOIN/"

from utils.data import data_loader
from utils.config import list_indicators
import numpy as np
import random

df_train,df_test=data_loader(path=data_dir + "Gemini_BTCUSD_1h.csv",train_test_split=0.9)

l_train = [i for i in range(0,len(df_train),24)]
l_test = [i for i in range(0,len(df_test),24)]

gt_train = []
gt_test = []
x_train = []
x_test = []
list_indicators.append("Close")

for i in range(len(l_train)-1):
    diff_train = df_train.iloc[l_train[i+1]-1]["Close"]-df_train.iloc[l_train[i]]["Close"]
    if diff_train>0:
        gt_train.append(0)
    else:
        gt_train.append(1)
    x_train.append([df_train.iloc[l_train[i]:l_train[i]+3][k] for k in list_indicators])
for j in range(len(l_test)-1):
    diff_test = df_test.iloc[l_test[j+1]-1]["Close"]-df_test.iloc[l_test[j]]["Close"]
    if diff_test>0:
        gt_test.append(0)
    else:
        gt_test.append(1)
    x_test.append([df_test.iloc[l_test[j]:l_test[j]+3][k] for k in list_indicators])


random_index_list = random.sample(range(0,len(gt_train)),int(0.2*len(gt_train)))
x_val = []
gt_val = []
gt_train_ = []
x_train_ = []
for i in range(len(x_train)):
    if i in random_index_list:
        x_val.append(x_train[i])
        gt_val.append(gt_train[i])
    else:
        x_train_.append(x_train[i])
        gt_train_.append(gt_train[i])
# CHECK
# print(len(x_val))
# print(len(gt_val))
# print(len(random_index_list))
# print(len(x_train_))
# print(len(gt_train_))
# print(len(x_train)-len(x_val))
x_train = np.array(x_train_)
x_val = np.array(x_val)
x_test = np.array(x_test)
gt_train = np.array(gt_train_)
gt_val = np.array(gt_val)
gt_test = np.array(gt_test)

np.save("{}/x_train.npy".format(path_dir+"/project_code/technical_analysis/data/"),x_train)
np.save("{}/x_val.npy".format(path_dir+"/project_code/technical_analysis/data/"),x_val)
np.save("{}/x_test.npy".format(path_dir+"/project_code/technical_analysis/data/"),x_test)
np.save("{}/gt_train.npy".format(path_dir+"/project_code/technical_analysis/data/"),gt_train)
np.save("{}/gt_val.npy".format(path_dir+"/project_code/technical_analysis/data/"),gt_val)
np.save("{}/gt_test.npy".format(path_dir+"/project_code/technical_analysis/data/"),gt_test)






    

