#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init('/home/michael/spark')

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.streaming import DataStreamReader
from textblob import TextBlob
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


# In[3]:


stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def clean_text(words):

    if words is None:
        return ''
    else:
        # remove URL from text
        #each_text_no_url = re.sub(r"http\S+", "", each_text)

        # remove numbers from text
        #text_no_num = re.sub(r'\d+', '', each_text_no_url)

        # tokenize each text
        #word_tokens = [str(i) for i in word_tokenize(each_text)]

        # remove sptial character
        clean_text = []
        for word in words:
            clean_text.append("".join([e for e in word if e.isalnum()]))

        # remove stop words and lower
        text_with_no_stop_word = [w.lower() for w in clean_text if not w in stop_words]  

        # do stemming
        stemmed_text = [stemmer.stem(w) for w in text_with_no_stop_word]

        array = " ".join(" ".join(stemmed_text).split())
        words = [str(i) for i in array]
        return words

cleanText_udf = udf(lambda text: clean_text(text), ArrayType(ArrayType(StringType())))


# In[4]:


embedding_dim = 100
sequence_len = 100

with open('tk.pickle', 'rb') as inFile:
    tokenizer = pickle.load(inFile)
    #tk_config = json.dumps(tk_config)
    #tokenizer = tokenizer_from_json(tk_config)
    
def tokenize(text):
    tokens = pad_sequences(tokenizer.texts_to_sequences(text),maxlen=sequence_len)
    return tokens

tokenizer_udf = udf(lambda text: tokenize(text), ArrayType(IntegerType()))


# In[5]:


model = load_model('model-lstm.h5')

def make_pred(words):
    print(words.shape)
    return model.predict(words)

predict_udf = udf(lambda word_vec: word_vec.shape)


# In[6]:



def preprocessing(lines):
    words = lines.withColumn('split_rows', split(lines.value, '_row_end').getItem(0)).drop('value')
    words = words.withColumn('word', split(words.split_rows, '_split_').getItem(0))                .withColumn('location', split(words.split_rows, '_split_').getItem(1))                            .drop('split_rows')
    words = words.na.replace('', None)
    #words = words.na.drop()
    words = words.withColumn('word', F.regexp_replace('word', r'http\S+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '@\w+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '#', ''))
    words = words.withColumn('word', F.regexp_replace('word', 'RT', ''))
    words = words.withColumn('word', F.regexp_replace('word', ':', ''))
    return words

def clean_tokenize_predict(lines):
    words = lines.withColumn('split_rows', split(lines.value, '_row_end').getItem(0)).drop('value')
    words = words.withColumn('word', split(words.split_rows, '_split_').getItem(0))                .withColumn('location', split(words.split_rows, '_split_').getItem(1))                            .drop('split_rows')
    words = words.na.replace('', None)
    words = words.withColumn('word', F.regexp_replace('word', r'http\S+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '@\w+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '#', ''))
    words = words.withColumn('word', F.regexp_replace('word', 'RT', ''))
    words = words.withColumn('word', F.regexp_replace('word', ':', ''))
    words = words.withColumn('clean_words', cleanText_udf(words.word))
    words = words.withColumn('tokens', tokenizer_udf(words.clean_words))
    words = words.withColumn('Pred', predict_udf(words.tokens))
    
    return words
    


# In[7]:


if __name__ == '__main__':
    #create spark session
    spark = SparkSession.builder.appName('TwitterSentimentAnalysis').getOrCreate()
    #spark.setLogLevel('ERROR')
    
    
    #read data from socket
    
    data = spark.readStream.format('socket').option('host','127.0.0.1').option('port', 5555).load()
    
    df = clean_tokenize_predict(data)

    
    query = df.writeStream.outputMode('append')                        .format('console')                        .trigger(processingTime='5 seconds').start()

    query.awaitTermination()


# In[ ]:




