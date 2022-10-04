#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init('/home/michael/spark')

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
import pyspark.sql.types as tp
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql import functions as F
import pandas as pd


# In[2]:


sc = SparkContext(appName="PySparkShell")
spark = SparkSession(sc)
sc.setLogLevel('ERROR')


# In[3]:


schema = tp.StructType([
  tp.StructField(name= 'id',          dataType= tp.IntegerType(),  nullable= True),
  tp.StructField(name= 'keyword',       dataType= tp.StringType(),  nullable= True),
  tp.StructField(name= 'location',       dataType= tp.StringType(),   nullable= True),
  tp.StructField(name= 'text',       dataType= tp.StringType(),   nullable= False),
  tp.StructField(name= 'target',       dataType= tp.IntegerType(),   nullable= False),  
])

train_data = spark.read.csv('train.csv', schema=schema, header=True)

# view the data
train_data.show(5)

# print the schema of the file
train_data.printSchema()


# In[4]:


def filter_empty(l):
    return filter(lambda x: x is not None and len(x) > 0, l)

filter_empty_udf = udf(filter_empty, tp.ArrayType(tp.StringType()))


# In[5]:


tokenizer = RegexTokenizer(inputCol="text", outputCol='tokens', pattern='\\W')
word_filter = StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')
vectorizer = Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize= 10, seed=42)


# In[6]:


token_data = tokenizer.transform(train_data)
token_data.show(5)


# In[7]:


nona_tokens = token_data.select(F.array_remove(token_data.tokens, '')).collect().alias(nona_tokens)

#nona_words = token_data.select(filter_empty_udf(F.col('tokens'))).alias('nona_words')
#nona_words.show(5)

nona_tokens.show(5)


# In[ ]:


filt_data = word_filter.transform(token_data)
train_nona_words = filt_data.select(filter_empty_udf('filtered_words')).alias('nona_words')
train_nona_words.show(5)


# In[ ]:


train_data = train_data.withColumn('filt_nona_words', train_nona_words.select(F.col('nona_words')))
                                        
train_data = train_data.drop('keyword').drop('location').drop('text').drop('tokens')

train_data.show(5)


# In[ ]:


vec_model = vectorizer.fit(train_data_filt_drop)
vec_model.getVectors.show(5)


# In[ ]:


train_data = vectorizer.transform(train_data)


# In[ ]:





# In[ ]:




