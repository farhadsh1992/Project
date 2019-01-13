#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 11:56:55 2019

@author: Farhad
"""

from sklearn.model_selection import KFold
import pandas as pd

class split():
    
    def __init__(self):
        self.trainDF = pd.DataFrame()
        self.validateDF = pd.DataFrame()
        
        
    def SplitKFold(self,,df, spnumber):
        kf = KFold(spnumber)
        fold=1
        for train_index, validate_index in kf.split(df):
            self.trainDF = pd.DataFrame(df.iloc[train_index, :])
            self.validateDF = pd.DataFrame(df.iloc[validate_index])
            print(f"""Fold #{fold}, Training Size: {len(trainDF)}, 
                  Validation Size: {len(validateDF)}""")
            fold += 1
        