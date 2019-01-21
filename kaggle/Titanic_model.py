#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:01:33 2019

@author: Farhad

For : Titanic data
"""

import pandas as pd
import numpy as np

from farhad_DL.utility import missing_median, missing_default,missing_maxitration
import re




def modify_Fare(train, test):
    data = [train, test]
    for dataset in data:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
        dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
        dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
        dataset['Fare'] = dataset['Fare'].astype(int)
    return train, test

def modify_Embarked(train, test):
    ports = {"S": 0, "C": 1, "Q": 2}
    train.Embarked = train.Embarked.map(ports)
    test.Embarked = test.Embarked.map(ports)
    return train, test


def modify_sex(train, test):
    genders = {'male':0, "female":1}
    train.Sex = train.Sex.map(genders) 
    test.Sex = test.Sex.map(genders) 
    
    return train, test

def modify_name(train, test):
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    
    data = [train, test]
    for dataset in data:
        # extract titles
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        # replace titles with a more common title or as Rare
        dataset['Title'] = dataset['Title'].replace(['Lady', 
                                                      'Countess','Capt', 
                                                     'Col','Don', 'Dr','Major', 
                                                     'Rev', 'Sir', 'Jonkheer', 'Dona'],
                                                    'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        
        # convert titles into numbers
        dataset['Title'] = dataset['Title'].map(titles)
        # filling NaN with 0, to get safe
        dataset['Title'] = dataset['Title'].fillna(0)
    train = train.drop(['Name'], axis=1)
    test = test.drop(['Name'], axis=1)
    
    return train, test


def missing_random():
    data = [train_df, test_df]
    for dataset in data:
        mean = train_df["Age"].mean()
        std = test_df["Age"].std()
        is_null = dataset["Age"].isnull().sum()
        # compute random numbers between the mean, std and is_null
        rand_age = np.random.randint(mean - std, mean + std, size = is_null)
        # fill NaN values in Age column with random values generated
        age_slice = dataset["Age"].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        dataset["Age"] = age_slice
        dataset["Age"] = train_df["Age"].astype(int)
    print(train_df["Age"].isnull().sum())
    return train_df, test_df


def modify_cabin(train, test):
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    data = [train,test]
    for dataset in data:
        dataset['Cabin'] =dataset['Cabin'].fillna('U0')
        dataset['Deck'] = dataset["Cabin"].map(
            lambda x : re.compile('([a-zA-z]+)').search(x).group())
        dataset['Deck'] = dataset['Deck'].map(deck)
        dataset['Deck'] = dataset['Deck'].fillna(0)
        dataset['Deck'] = dataset['Deck'].astype(int)
        
def modify_age(train, test):
    data= [train, test]
    for dataset in data:
        dataset['Age'] = dataset['Age'].astype(int)
        dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
        dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
        dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
        
    return train, test
def make_relative(train, test):
    data = [train, test]
    for dataset in data:
        dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
        dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
        dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
        dataset['not_alone'] = dataset['not_alone'].astype(int)
    return train, test

def Fare_per_Person(train, test):
    data = [train, test]
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
    return train, test
def data_modify(train, test):
    train = train.drop(['PassengerId',"Ticket"],axis=1)
    test = test.drop(['Ticket'],axis=1)
    
    # Cabin
    modify_cabin(train, test)
    train = train.drop(['Cabin'], axis=1)
    test = test.drop(['Cabin'], axis=1)
    
    # Age:
    missing_median(train,'Age')
    missing_median(test,'Age')
    train, test = modify_age(train, test)
    
    # Embarked
    missing_maxitration(train,'Embarked')
    missing_maxitration(test,'Embarked')
    train, test = modify_Embarked(train, test)
    
    # Fare
    train['Fare'], test['Fare'] = train["Fare"].fillna(0),test['Fare'].fillna(0)
    train['Fare'],test['Fare'] = train["Fare"].astype(int),test['Fare'].astype(int)
    
    # Name
    train, test = modify_name(train, test)
    
    #Sex 
    train, test = modify_sex(train, test)
    
    
    # Fare
    train, test = modify_Fare(train, test)
    
    # Creating new Features
    train, test = make_relative(train, test)
    train['Age_Class'] = train.Age * train.Pclass
    test['Age_Class'] = test.Age * test.Pclass
    
    train, test = modify_age(train, test)
    
    return train, test

def model_ML(model, x_train, x_test, y_train, y_test):
    
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    pred = np.round(pred)
    
    pred_train = model.predict(x_train[:40])
    pred_train = np.round(pred_train)
    
    score = (accuracy_score(y_test, pred), mean_absolute_error(y_test, pred),
             round(model.score(x_test, y_test), 3))
    sce = confusion_matrix(y_test, pred)
    
    score_train = (accuracy_score(y_train[:40], pred_train), mean_absolute_error(y_train[:40], pred_train),
                  round(model.score(x_train, y_train), 3))
    return score,score_train, sce


def draw_model(title, model):
    score_test[title], score_train[title], sce = model_ML(model,x_train, x_test, y_train, y_test)
    print("Final Score of train:",score_train[title])
    print("Final Score of test:",score_test[title])
    plot_confusion_matrix(sce, ['dead','live'], title)