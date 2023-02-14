from cProfile import label
from turtle import color
import pandas as pd
from ta import add_all_ta_features
import numpy as np 
from talib import abstract
# from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

def data_loader(path,train_test_split=0.8):
    df = pd.read_csv(path)
    df = df.iloc[::-1]
    df.rename(columns={"date":"Date","close":"Close","open":"Open","high":"High","low":"Low"},inplace=True)
    df = df.reset_index()
    df.drop(["index"],axis=1,inplace=True)
    #all technical indactors
    df = add_all_ta_features(df,open="Open", high="High", low="Low", close="Close", volume="Volume USD",fillna=True)
    df_unorm = df.copy()
    # for i in df.columns:
    #     df[i] = (df[i]-df[i].min())/(df[i].max()-df[i].min())
    #split data train/test
    l = len(df)
    train_test =int(l * train_test_split)
    df_train = df[0:train_test]
    df_train.drop(["unix","Date","symbol"],axis=1,inplace=True)
    df_test = df[train_test:]
    df_test = df_test.reset_index()
    df_test.drop(["index","unix","Date","symbol"],axis=1,inplace=True)
    unormalized_close_train = df_train["Close"].copy()
    unormalized_close_test = df_test["Close"].copy()
    train_returns = [0 if i==0 else df_train["Close"][i]-df_train["Close"][i-1] for i in range(len(df_train["Close"]))]
    test_returns = [0 if i==0 else df_test["Close"][i]-df_test["Close"][i-1] for i in range(len(df_test["Close"]))]
    unormalized_rsi_train = df_train["momentum_rsi"].copy()
    unormalized_rsi_test = df_test["momentum_rsi"].copy()
    for i in df_train.columns:      
        df_test[i] = (df_test[i]-df_unorm[i].min())/(df_unorm[i].max()-df_unorm[i].min())
        df_train[i] = (df_train[i]-df_unorm[i].min())/(df[i].max()-df_unorm[i].min())
    df_train["Returns"] = train_returns
    df_test["Returns"] = test_returns
    df_train["Unorm_Close"] = unormalized_close_train
    df_test["Unorm_Close"] = unormalized_close_test
    df_train["Unorm_rsi"] = unormalized_rsi_train
    df_test["Unorm_rsi"] = unormalized_rsi_test
    

    return df_train,df_test
    






