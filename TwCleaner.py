#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 11:06:51 2019

@author: Farhad
"""

import pendulum
import re
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.tokenize import WordPunctTokenizer
import sys
import psutil
import os
import multiprocessing
import pandas as pd
from farhad.preTexteditor import editor
from numba import jit
import string
from nltk.stem import SnowballStemmer
from farhad.time_estimate import EstimateFaster
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

print("------------------------------------------------------------------------")
print("Class:  Tweets_preprocesiing ")
print("")
print("input: df")
print("Function:")
print("          CT = Tweets_preprocesiing()")
print("          data1 = CT.Cleaner(df)")
print("          new_data = Remove_stop_words(data1)")
print("          newdf = save_clean_tweets(df,new_data,created_at='created_at',name='clean.csv')")
print("")
print(" ")
print("------------------------------------------------------------------------")


class Tweets_preprocesiing():
    #def __init__(self):
        #self.df = df
        #self.new_list = ['' for i in  range(len(self.df))]
    
    def Remove_stop_words(self,df,title='Remove stop_words'):
    
        stop_words = [x for x in stopwords.words('english') if x!='not'] 
        
        new_list2=[]
        for num,text in enumerate(df):
            filtered = []
            text_token = word_tokenize(str(text))
            for w in text_token:
                if w not in stop_words:
                    filtered.append(w)
            new_list2.append(" ".join(filtered).strip())
            EstimateFaster(num,df,title)
             
        print(" *** Done! ***")   
        return new_list2

    def TwitterCleaner(self,text):
        stem = SnowballStemmer('english')
        
        tok = WordPunctTokenizer()
        pat1 = r'@[A-Za-z0-9_]+'
        pat2 = r'https?://[^ ]+'
        www_pat = r'www.[^ ]+'
        part3 = string.punctuation # remove 's
        pat4 = r'@[\s]+'
        combined_pat = r'|'.join((pat1, pat2))

        negations_dic = {"isn't":"is not", "aren't":"are not", 
                         "wasn't":"was not", "weren't":"were not",
                         "haven't":"have not","hasn't":"has not",
                         "hadn't":"had not","won't":"will not",
                         "wouldn't":"would not", "don't":"do not",
                         "doesn't":"does not","didn't":"did not",
                         "can't":"can not","couldn't":"could not",
                         "shouldn't":"should not","mightn't":"might not",
                         "mustn't":"must not","isnt":"is not", "arent":"are not", 
                         "wasnt":"was not", "werent":"were not",
                         "havent":"have not","hasnt":"has not",
                         "hadnt":"had not","wont":"will not",
                         "wouldnt":"would not", "dont":"do not",
                         "doesnt":"does not","didnt":"did not",
                         "cant":"can not","couldnt":"could not",
                         "shouldnt":"should not","mightnt":"might not",
                         "mustnt":"must not","ist":"is not", "aret":"are not", 
                          
                         "havet":"have not","hasnt":"has not",
                         "hadnt":"had not","wont":"will not",
                         "wouldt":"would not", "dont":"do not",
                         "doest":"does not","didt":"did not",
                         "cant":"can not","couldnt":"could not",
                         "shouldt":"should not"}
        neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
        
        text = re.sub(combined_pat,'',text)
        soup = BeautifulSoup(text, 'lxml')
        souped = soup.get_text()
        
        try:
            bom_removed = souped.decode("utf-8").replace(u"\ufffd", "?")
        except:
            bom_removed = souped
            
        lower_case = bom_removed.lower()
        #lower_case = stem.stem(lower_case)
        stripped = re.sub(www_pat, '', lower_case)
        stripped = re.sub(pat4,'',stripped)
        stripped = re.sub(combined_pat, '', stripped)
        stripped = re.sub('https ', '', stripped)
        stripped = re.sub('rt ', '', stripped)
        stripped = re.sub("[^a-zA-Z]", " ", stripped) # only letter
        
        
        words = [stem.stem(word) for word in tok.tokenize(stripped)]
        #neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], stripped)
        #stripped = re.sub("[^a-zA-Z]", " ", stemmed_words)
        
        # During the letters_only process two lines above, it has created unnecessay white spaces,
        # I will tokenize and join together to remove unneccessary white spaces
        #words = [x for x  in tok.tokenize(stemmed_words) if len(x) > 1]
        if len(words)>2:
            return (" ".join(words)).strip()
        else:
            return None
    
    def Cleaner(self,df):
        new_list=[]
        for num,text in enumerate(df):
            new_list.append(self.TwitterCleaner(text))
            EstimateFaster(num,df,'clean tweets')
        
        print(" *** Done! ***")
        return new_list
    def save_clean_tweets(df,new_data,created_at='created_at',name="clean.csv"):
        
        new_df = pd.DataFrame()
        new_df[created_at] = df[created_at]
        new_df['text'] = new_data
        new_df.to_csv(name,index="False")
        print("*** new_dataframe_save! :)***")
        return new_df
        