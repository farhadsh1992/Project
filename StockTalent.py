#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:49:50 2019

@author: Farhad
"""
import pandas as pd
from farhad.time_estimate import EstimateFaster

def normalize_df(df):
    df_norm = (df - df.mean()) / (df.max() - df.min())
    return df_norm

def Normlize_nMonth(numday=90,df,Date="Date",price="Adj Close"):
    #df['month'] = df[Date].apply(lambda x: datetime.datetime.strptime(str(x),"%Y-%m-%d").month)

    lenght = int(round(len(df[price])/numday))-1
    Plist = [[] for i in range(lenght)]
    new_df = pd.DataFrame()
    for num  in range(0,lenght):
        Plist[num] = [x for x in (df[price][numday*num:(numday*(num+1))])]
        new_df[str(num)] = Plist[num] 
        EstimateFaster(num,Plist[num],'time')
        new_df['norm_{}'.format(num)] = normalize_df(new_df[str(num)])
        
    Plist2 = pd.Series([x for x in (df[price][numday*lenght:])])
    Plist2 = normalize_df(Plist2)
    frames = [new_df['norm_{}'.format(num)] for num in range(lenght)]
    x = pd.concat(frames,axis=0)
    frames = [x, Plist2]
    result = pd.concat(frames, names=['index','norm']).sample(frac=1).reset_index(drop=True)
    print('*** Done! ***')
    return  result

def Extract_month(x):
     return datetime.datetime.strptime(str(x),"%Y-%m-%d").month
 
def label_according_model_one(df,target="norm_Adj_price"):
    def Give_lebal_according_price(x):
        if x>=0.2:
            y=2
        elif x>=0:
            y=1
        elif x<0:
            y=-1
        elif x<=-0.2:
            y=-2
        return y
    df["Price_label"] = df[target].apply(Give_lebal_according_price)
    
    return df
def label_according_model_two(df,target="norm_Adj_price"):
    def Give_lebal_according_price(x):
        
        elif x>=0:
            y=1
        elif x<0:
            y=o
        
        return y
    df["Price_label_(0,1)"] = df[target].apply(Give_lebal_according_price)
    
    return df