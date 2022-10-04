Problem Statement and Background
Our select project is to recognize disasters using the Twitter platform. The project aims to build a data pipeline that finds and identifies tweets about disasters, earthquakes, tornados, floods, etc. To complete this task we will utilize a pre-existing dataset from Kaggle titled “Natural Language Processing with Disaster Tweets”, which was initially created by the company Figure Eight [1]. The dataset is split into two comma-separated values files – a train and test dataset. These sets include the following features an integer ID, a keyword from that tweet, the location the tweet was sent from, the text of a tweet, and finally, the target that is the different classes of whether the tweet corresponded to a disaster or not, only included in the training dataset. An essential characteristic of this dataset is that the included attributes or features can be pulled using Tweeter Developer API, with the exception of the target variable which we are aiming to predict in our analysis [2]. 
Some of the success metrics central to our analysis include accuracy, precision, recall, and finally, f-score. There are accuracy or quality measure which fit the nature of our project. These metrics were chosen as they are central in classification problems.  We are aiming for a model where we do not mind a high rate of false positives. On the other hand, we must avoid a high rate of false negatives, as this would mean tossing out the occasional tweets that could be identified as a new disaster report. 
The motivation of this project is to minimize the time it takes to notify first response teams, government agencies of disasters. The sooner these institutions are aware of a disaster, the quicker they can provide aid. Faster response times will likely reduce fatalities and injuries during these events. The apparent impetus for this project is to lessen human suffering. In the smartphone era, most people carry a device in their pocket to share data on the internet. Though it may be inadvertent, this has crowd-sourced disaster detection essentially. In many situations, this dispersed network of observers can identify and report disasters faster than official channels of government agencies. This information exists on the web, and there exists an opportunity to leverage it to provide faster aid to people caught in catastrophes. It would be a shame to allow this data to go unused. We hope that our project helps to fill this hole.
 
 
Methods
Data Organization
 
In terms of data organization, the first task was to do some exploratory data analysis which included several visualizations in order to better understand the dataset, these were done using Matplotlib and Plotly's Python graphing libraries. 
 
The next task was data preprocessing which began with cleaning the data to get it in a form that the chosen models could better understand and more importantly can handle. Our data cleansing process included removing special characters, numbers, URLs, HTML, and stopwords which are words in each tweet which does not significant meaning, and finally stemming each for of the text to its root word. These cleaning procedures were applied to the test and train datasets.
 
Another issue our initial data exploration of the Tweet Disaster dataset found was that the data had lots of missing entries especially for the keyword and location columns. This challenge was not something we anticipated, but we found it was a common problem when dealing with social media data. To address the issue, we looked at a few different solutions, namely, we could omit the missing entries, or we could impute the missing values. The imputation techniques we explored included forward fill, where null entries are replaced with the last observed value, backfill, where null entries are replaced with the next observed value. However, given that each Tweet is not related to the next as in time-series datasets we chose to impute each null keyword entry with a value of none. 
 
Next, our group addressed feature creation, which is the process of constructing new features from existing data in order to best train our selected models. As we wanted to include as many features into the analysis as possible we chose to combine the tweet keyword and text features. The keyword and text values were concatenated into a new aggregate feature. This aggregate feature is the core of our analysis as we aim to build a model to mainly analyze the content of a tweet. The one major downside of this approach is that our predictive model would miss out on the many tweets that might stem from one location during a disaster. However, we found this approach sufficient for the context of this problem and could be addressed upon future iterations. 
 
Our group then moved on to the methods to feed this new aggregate feature we engineered into each model. Techniques including such as vectorization tokenization, padding, and word embedding were employed depending on the model algorithm. 
 
For the first model Gradient Boosting Classifier (GBC) discussed in the modeling algorithms section, vectorization, that is, the mapping of words to vectors of real numbers was conducted through Sklearn’s CountVectorizer method. For the other chosen models namely Long Short Term Memory (LSTM) and Gated Recurrent Unit (GRU), we tokenized the data using TensorFlow’s Tokenizer class. This class allows vectorizing a text corpus, by turning each text into either a sequence of integers, where each integer is the index of a token in a dictionary. The Tokenizer class is based on TF-IDF [n+1].
 
We then used Global Vectors (GloVe) embedding, as GloVe captures both global statistics and local statistics of a corpus, in order to come up with word vectors[6]. We chose a 100 dimension dataset. After we read the GloVe embedding file then we create an embedding layer using Keras. This leads to padding methods namely TensorFlow’s pad_sequences function. This function transformed our list of Tweet text sequences into a 2-D NumPy array. Finally, we used the train_test_split() method from Sklearn to split the dataset into a 70:30 ratio train and test, respectively. 
 
Querying
Inspired by our assignment from earlier in the semester, we used Twitter’s developer API to stream in Tweets from the web based on the dataset’s unique keywords. We used the Tweepy module in Python and created a filtered streamed based on a keyword set taken from the training data. Then the search results were reduced to include the text of the tweet, the user location, and coordinates. The collected results were sent over a TCP socket to the next stage of the pipeline
Initially we attempted to use the Spark Structured Streaming API which is an upgrade from the original Spark streaming. The Structured Streaming received the search results from the socket and handles data in a DataFrame format. This was appealing for a number of reasons. Firstly, it is optimized and therefore faster than the original Spark streaming. The use of DataFrames lends itself to the processing of data in batches. Since our task involves repeating the same operations on every Tweet that is returned in the search, this seemed ideal. Additionally, the collection of results is built into Structured Streaming. Calling functions to operate on columns produces code that is easy to understand and intuitive. 
Unfortunately, Spark Structured Streaming did not cooperate with Tensorflow and Keras for our project. In particular, the Java code which is at the center of Spark did not work well to load our trained tokenizers and prediction model. There were a number of errors concerning datatypes and loading pretrained functions that we could not resolve. The tokenizers and models worked just fine in python notebooks, but could not be integrated with Spark’s Structured Streaming. 
Instead, we settled on RDD streaming with Spark. It is a bit clunkier and not as sleek as the Structured Streaming, but it gets the job done. With Spark Streaming we received the search results from the Twitter App, preprocessed the texts, and used the trained model to make predictions on each of the tweets. When the model flagged a tweet that was a potential disaster, it printed the message to the console.  
 
 
Modeling Algorithms
To begin we wanted to start our classification using a classic machine learning modeling algorithm and after much research, we ended up choosing the Gradient Boosting Classifier (GBC) from Sklearn. Gradient boosting machines are powerful ensemble machine learning algorithm that uses decision trees. Where boosting is a general ensemble technique that involves sequentially adding models to the ensemble where subsequent models correct the performance of prior models [4]. The GBC is especially useful for classification tasks, which makes it well-suited for our problem. One benefit of the GBC model is that variations in model performance are addressed through parameter tuning rather than altering the model’s structure.
Next, we applied LSTM learning algorithms, specifically Bidirectional Long Short Term Memory (BLSTM). LSTM is a kind of Recurrent Neural Network (RNN) that is capable of learning long-term dependencies and they can remember information for a long period of time as they designed with an internal memory system. LSTM facilitated us to give a sentence as an input for prediction rather than just one word, which is much more convenient in NLP and makes it more efficient. For the LSTM model, the structure and activation function played a bigger part in the model’s performance. 
This leads nicely to our last choice for the chosen learning algorithms namely Gated Recurrent Unit (GRU). GRU was chosen as it reduces the number of parameters from LSTM. In GRU, there is no explicit memory unit. The memory unit is combined along with the network. There is no forget gate and update gate in GRU. They are both combined together and thus the number of parameters is reduced.
Other Methods and Issues
One issue that arose was the method by which we saved the model’s weights to be processed by Spark Stream. Initially, we saved them as a JSON file however this gave several errors when the data was subsequently prepossessed in the Virtual Box environment.  
Although the project functions in that it searches tweets and makes predictions on which ones are about real disasters, we found that most of the tweets returned from the search did not include any location information. Twitter user’s have the option to provide their location when they post a tweet, and can even provide exact longitude and latitude coordinates. However, this feature is rarely used. This brings up questions about the practicality of the project. Even when tweets are correctly classified, it is often the case that there was no location data attached to them. A solution might be to build a further system that can make predictions on the location based on information provided in a tweet. However, this was outside the scope of this project. It also raises difficult questions as to the ethics of using Machine Learning and Data Science tools to track the location of people without their consent.
If this project continues it would be interesting to explore the use of Spark Streaming and Kafka. Although we did not have time to implement such a system, the tools for such an endeavor exist. Kafka producer-consumer scheme could be useful to have multiple nodes searching for tweets and a distributive system to process and make predictions on those tweets. 
 
Lessons Learned
The project exposed our team to a number of Big Data tools. We also learned a great deal through the numerous challenges we encountered. We explored the power, and difficulty, of combining a number of tools in a data pipeline. Sklearn and NLTK were used in the preprocessing of the training data. Tnensorflow’s Keras was leveraged to develop and train a model to make real-time predictions on tweets. We used the Twitter API and Tweepy module to filter and stream tweets over a local socket. Matplotlib and Plotly provided us with visualizations for our data and results. The Pandas module was critical in a number of steps for managing the data. Finally, we used Spark Streaming and Pyspark to put all of the pieces together. 
This project taught us the importance of data preprocessing. We spent as much time in energy getting the data into the correct form for our tools as we did develop the actual tools. Also, the lack of location data in our tweets underscores the importance of data collection and mining as an essential first step to any data pipeline. Although Spark Structured Streaming was not, in the end, fruitful for this project, wrestling with it allowed us to learn many things about the engine and prepared us to leverage it again in the future. 
 
 
 
 
 
 
 
 
 
 
 
 
 
Results
Table 1.0:  Model metric results including accuracy, precision, recall, and f1-score for both training and test sets.


Model

Accuracy

Precision

Recall

F1-Score
Train
Test
Train
Test
Train
Test
Train
Test
GBC1
77.52%
75.22%
91.96%
84.05%
53.12%
49.58%
67.34%
62.37%
GBC2
98.25%
78.28%
98.99%
78.28%
96.99%
69.34%
97.98%
73.54%
GBC3
91.86%
80.30%
 95.74%
81.23%
85.12%
68.18%
90.12%
74.14%
LSTM1
96.56%
77.75%
98.27%
75.33%
93.93%
67.65%
95.92%
70.57%
LSTM2
98.63%
78.19%
99.52%
78.09%
97.29%
64.14%
98.33%
69.59%
LSTM3
97.69%
77.97%
98.46%
75.91%
96.17%
69.07%
97.19%
69.07%
GRU
83.16%
78.02%
96.78%
85.92%
64.02%
56.06%
76.18%
66.73%

 
The seven models varied in overall performance and their results are displayed in Table 1.
For GBC1 the default parameter values were used to define the model. GBC1 had an accuracy of 77.52% on the test set, thus we were off to a great start and resulted in a precision of 91.96%.
When defining GBC2 the main difference was the increased number of estimators by a factor of 200. The number of estimators is defined as the number of boosting stages performed during training. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance as was the same in our case as the accuracy increased from 77.52% to 98.25% after the increase. Thus, it is clear that GBC2 outperformed GBC1; however, such a large increase in the metric evaluation lead us to believe that the model was in fact overfitting.
For GBC3 we ran several iterations varying different params. This lead to the use of a configuration using a collection of parameters such as max_depth, min_samples, min_split, etc. This resulted in the best-performing and more important better consistently performing model for both the train 91.86% and test 80.30%. Thus GBC3 was used as the baseline to compare the other models utilizing the LSTM learning algorithm, discussed in the next paragraph.
For LSTM1 was five layers, the first a 1D convolution layer which employed a rectified linear activation function (ReLU), then a Bidirectional LSTM layer with 128 nodes or neurons, followed by two dense layers with 256 and then 128 layers respectively both using ReLU activation and the final output layer of one neuron used a sigmoid activation function. 
The model used the Adam optimizer with a loss function of binary cross-entropy, a learning rate of 1e-3, and was trained for 20 epochs. LSTM1’s results were quite similar to GBC2 that appeared to be overfitting the data. Specifically, it performed very well on the training accuracy of 96.56% but not very well on the test set of 77.75%.
LSTM2 configuration was similar to that of LSTM1 with the major addition of adding dropout laters and batch normalization in order to mediate overfitting. A dropout of 50% was added to the BLSTM layer a well as between each dense layer. Again The model used the Adam optimizer with a loss function of binary cross-entropy, a learning rate of 1e-3, and was trained for 50 epochs. There was little improvement from LSTM1 yet some minor improvement in terms of metrics. 
In hopes to break away from overfitting for LSTM3 model had a different structure. Instead of the first layer being a 1D convolution layer we defined a sequential model with fewer layers and more dropout layers between them. There was little improvement from LSTM1 yet some minor improvement in terms of metrics. 
Finally, GRU as previously mentioned the GRU algorithm has fewer parameters to tune as thus was a reasonable choice at this point in our analysis. GRU performed similarly to previous iterations. 
 
 
 
Big Data Systems and Tools
All of the codings for this project were done in Jupyter Notebook either on our local machine or through a Virtual Box running some version of Ubuntu (20.0.4 and 16.0.4). 
As far as big data analysis and visualization systems we employed Pandas and Matplotlib. From big data modeling, machine learning, and deep learning systems we used TensorFlow, Scikit-learn, and Keras. Lastly from big data stream and time-series processing systems, our group looked into Spark Streaming.
One idea our group did have was to attempt to store tweets in No-SQL such as Mongo DB. This would have been useful if we streamed a large number of tweets using Spark then these tweets could have been stored and used later to train future model iterations. However, given the large hurdle to set up Spark and the little benefit until the project scaled we decided against this idea.
 
 
References
Natural language processing with disaster tweets. Kaggle. (n.d.). Retrieved December 10, 2021, from https://www.kaggle.com/c/nlp-getting-started/overview. 
Twitter. (n.d.). Use cases, tutorials, & documentation | twitter developer platform. Twitter. Retrieved December 10, 2021, from https://developer.twitter.com/en. 
Apache Spark. Downloads | Apache Spark. (n.d.). Retrieved December 10, 2021, from http://spark.apache.org/downloads.html. 
Sklearn.ensemble.gradientboostingclassifier. scikit. (n.d.). Retrieved December 10, 2021, from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html. 
Ganegedara, T. (2021, November 15). Light on math ML: Intuitive Guide to Understanding Glove embeddings. Medium. Retrieved December 10, 2021, from https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010. 
 
 
 
 
