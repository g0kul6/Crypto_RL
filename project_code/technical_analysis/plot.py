
import os
import sys
path_dir = os.path.abspath(os.getcwd())
sys.path.append(path_dir+"/project_code")
data_dir = path_dir + "/dataset/BITCOIN/"
from sklearn.metrics import confusion_matrix
from utils.data import data_loader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


gt = np.load("/home/g0kul6/g0kul6/RL-PROJECT/Project_RL/project_code/technical_analysis/outputs/gt_mlp_lr-0.0001_batch_size-32_epoch-1000.npy")
output = np.load("/home/g0kul6/g0kul6/RL-PROJECT/Project_RL/project_code/technical_analysis/outputs/output_mlp_lr-0.0001_batch_size-32_epoch-1000.npy")

# cm = confusion_matrix(gt,output)

# df_cm = pd.DataFrame(cm,index = [i for i in "AB"],columns=[i for i in "AB"])
# plt.figure(figsize=(10,7))
# sn.heatmap(df_cm ,annot=True)
# plt.show()



df_train,df_test=data_loader(path=data_dir + "Gemini_BTCUSD_1h.csv",train_test_split=0.9)

l_train = [i for i in range(0,len(df_train),24)]
l_test = [i for i in range(0,len(df_test),24)]

plt.plot(range(len(df_test["Close"]-1)),df_test["Close"])
ups_index = []
ups_value = []
downs_index = []
downs_value = []
wrong_index = []
wrong_value = []
starts = []
starts_index = []
count = 0
for i in range(0,len(l_test)-1):
    if output[i]==0 and gt[i]==0:
        ups_index.append(l_test[i]+23)
        ups_value.append(df_test.iloc[l_test[i]+23]["Close"])
        starts.append(df_test.iloc[l_test[i]+3]["Close"])
        starts_index.append(l_test[i]+2)
        count = count + 1
    elif output[i]==1 and gt[i]==1:
        downs_index.append(l_test[i]+23)
        downs_value.append(df_test.iloc[l_test[i]+23]["Close"])
        starts.append(df_test.iloc[l_test[i]+2]["Close"])
        starts_index.append(l_test[i]+2)
        count = count + 1
    # else:
    #     wrong_index.append(l_test[i])
    #     wrong_value.append(df_test.iloc[l_test[i]]["Close"])
    #     starts.append(df_test.iloc[l_test[i]-21]["Close"])
    #     starts_index.append(l_test[i]-21)



plt.scatter(ups_index,ups_value,color='r',label="up")
plt.scatter(downs_index,downs_value,color='g',label="down")
plt.scatter(starts_index,starts,color='k',label="start")
plt.title("accuracy:{}/{}".format(count,len(gt)))
plt.legend()
plt.show()

        
