#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import tweepy
import socket


# In[2]:


# loading authentication keys for twitter API
# twitter_keys.json needs to contain twitter app credentials
with open('twitter_keys.json') as file:  
    keys = json.load(file)
    
consumer_key = keys["API Key" ]
consumer_secret = keys["API Secret Key" ]
access_key = keys["Access Token" ]
access_secret = keys["Access Secret Token"]

#load keywords to search
with open("keywords.json") as file:
    file_data = json.load(file)
    
keywords = file_data['keywords']


# In[3]:


class tweet_stream(tweepy.Stream):
    def __init__(self, api_key, api_secret, access_key, access_secret, c_socket):
        super().__init__(api_key, api_secret, access_key, access_secret)
        self.client_socket = c_socket
        
    def on_data(self, data):
        try:
            info = json.loads(data)
            print('new tweet')
            
            #append tweet to the message to be sent
            if 'extended_tweet' in info:
                tweet = str(info['extended_tweet']['full_text'])
            else:
                tweet = str(info['text'])
                
            #append location to message to be sent
            if info['user']['location']:
                loc = str(info['user']['location'])
            else:
                loc = str('None')
            
            if info['geo'] is None:
                coords = str('None')
            elif info['geo']['coordinates'] is None:
                coords = str('None')
            elif info['geo']['coordinates']['coordinates'] is None:
                coords = str('None')
            else:
                coords = str(info['geo']['coordinates']['coordinates'])
                
            msg = str(tweet +' _split_ '+loc +' _split_ '+ coords+' _row_end').encode('utf-8')

            #send info
            self.client_socket.send(msg)
                    
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
            
        return True
        
    #def on_status(self, status):
        #print(status)
        
    def on_error(self, status):
        print('Error: {}'.format(status))

def sendData(c_socket):

    stream = tweet_stream(consumer_key,
                            consumer_secret,
                            access_key,
                            access_secret,
                            c_socket)
    
    stream.filter(track=keywords, languages=['en'])

if __name__ == "__main__":
    s = socket.socket()    # Create a socket object
    host = "127.0.0.1"     # Get local machine name
    port = 5556          # Reserve a port for your service.
    s.bind((host, port))   # Bind to the port

    print("Listening on port: %s" % str(port))

    s.listen(5)                 # Now wait for client connection.
    c, addr = s.accept() 
    
    print( "Received request from: " + str( addr ) )

    sendData(c)
    


# In[ ]:


#close the socket so it can be used in case of restart needed
s.close()


# In[ ]:




