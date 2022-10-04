import json
import tweepy
import socket

# loading authentication keys for twitter API

with open('twitter_keys.json') as file:
    keys = json.load(file)
    
consumer_key = keys["API Key" ]
consumer_secret = keys["API Secret Key" ]
access_key = keys["Access Token" ]
access_secret = keys["Access Secret Token"]


#define a tweet_stream class that is a subclass of the tweepy.Stream class
#it will access the twitter api and stream tweets

class tweet_stream(tweepy.Stream):
    def __init__(self, api_key, api_secret, access_key, access_secret, c_socket):
        super().__init__(api_key, api_secret, access_key, access_secret)
        self.client_socket = c_socket
        
    def on_data(self, data):
        try:
            tweet = json.loads(data)
            print( tweet['text'].encode('utf-8') )
            self.client_socket.send( tweet['text'].encode('utf-8') )
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
        
    def on_status(self, status):
        print(status)

        
def sendData(c_socket):

    stream = twitter_stream(consumer_key,
                            consumer_secret,
                            access_key,
                            access_secret,
                            c_socket)
    
    stream.filter(track=['news'])
    

if __name__ == "__main__":
    s = socket.socket()         # Create a socket object
    host = "127.0.0.1"     # Get local machine name
    port = 5555
                 # Reserve a port for your service.
    s.bind((host, port))        # Bind to the port

    print("Listening on port: %s" % str(port))

    s.listen(5)                 # Now wait for client connection.
    c, addr = s.accept() 
    
    print( "Received request from: " + str( addr ) )

    sendData(c)