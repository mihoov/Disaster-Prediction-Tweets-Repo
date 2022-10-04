#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark

findspark.init('/home/michael/spark') #Change to appropriate file path

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

import os
import json

import pandas as pd
import numpy as np
import re
import string
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


# In[2]:


sc = SparkContext("local[2]", "TweetClassifier")
sc.setLogLevel("ERROR")
ssc = StreamingContext(sc, 1)


# In[3]:



stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')


def clean_text(each_text):

    # remove URL from text
    each_text_no_url = re.sub(r"http\S+", "", each_text)
    
    # remove numbers from text
    text_no_num = re.sub(r'\d+', '', each_text_no_url)

    # tokenize each text
    word_tokens = word_tokenize(text_no_num)
    
    # remove sptial character
    clean_text = []
    for word in word_tokens:
        clean_text.append("".join([e for e in word if e.isalnum()]))

    # remove stop words and lower
    text_with_no_stop_word = [w.lower() for w in clean_text if not w in stop_words]  

    # do stemming
    stemmed_text = [stemmer.stem(w) for w in text_with_no_stop_word]
    
    return " ".join(" ".join(stemmed_text).split())

sequence_len = 100

with open('tk.pickle', 'rb') as inFile:
    tokenizer = pickle.load(inFile)
    #tk_config = json.dumps(tk_config)
    #tokenizer = tokenizer_from_json(tk_config)
    
def tokenize(text):
    sequence_len = 100
    tokens = pad_sequences(tokenizer.texts_to_sequences(text),maxlen=sequence_len)
    return tokens

model = load_model('model-lstm.h5')

def make_prediction(words):
    try:
        return model.predict(words)
    except ValueError as e:
        print('Error: {}'.format(e))
        print('Model failed to process input. Skipping')

def clean_tokenize_predict(tweet):
    for t in tweet:
        tweet_substr = t.split('_split_')
        #print('2')
        text = tweet_substr[0]
        #print('3')
        txt_clean = [clean_text(text)]
        #print('4')
        tweet_tokens = tokenize(txt_clean)
        #print('5')
        
        pred = make_prediction(tweet_tokens)
        
        print('POSSIBLE DISASTER: {} probability: {}'.format(t, pred))
        #print(text)
        #print(pred)
        #print('')


# In[4]:


lines = ssc.socketTextStream("localhost", 5556)

words = lines.map(lambda line: line.split('_row_end'))

words.foreachRDD(lambda rdd: rdd.foreach(clean_tokenize_predict))

ssc.start()             # Start the computation
ssc.awaitTermination() 


# In[ ]:




