import os
import sys
import tweepy
import time

'''
Documentation:
https://docs.tweepy.org/en/stable/

Source code:
https://github.com/tweepy/tweepy

Explanation step-by-step:
https://medium.com/@skillcate/python-tweepy-a-complete-guide-f0ff5ba54fce
https://medium.com/@skillcate/set-up-twitter-api-to-pull-2m-tweets-month-44d004c6f7ce

'''

# Recent search:
# https://github.com/xdevplatform/Twitter-API-v2-sample-code/blob/main/Recent-Search/recent_search.py

class TWITTER():
    def __init__(self, twitterEnv):
        __bearerToken = os.environ.get(twitterEnv)
        if not __bearerToken:
            raise EnvironmentError("TWITTER_TOKEN not found in Environment variables")

        self.twitterClient = tweepy.Client(bearer_token=__bearerToken)

    def getTweets(self, keyword: str, tweetFields: list[str], userFields: list[str]):
        tweets = []
        nextPage = None
        query = f"{keyword} lang:en"

        while True:
            time.sleep(1)
            try:
                response = self.twitterClient.search_recent_tweets(
                        query=query, 
                        tweet_fields=tweetFields,
                        max_results=50,
                        next_token=nextPage
                )

                for tweet in response.data:
                    print(f"Tweet ID: {tweet.id}")
                    print(f"Text: {tweet.text}")
                    print(f"Created At: {tweet.created_at}")
                    print(f"Author ID: {tweet.author_id}")
                    print(f"Public Metrics: {tweet.public_metrics}")
                    print(f"Language: {tweet.lang}")
                    print("-" * 50)

                    tweets.append(tweet)

                    user = self.twitterClient.get_user(id=tweet.author_id, user_fields=userFields)
                    print(f"User ID: {user.data.id}")
                    print(f"User Name: {user.data.name}")
                    print(f"Username: {user.data.username}")
                    print(f"Description: {user.data.description}")
                    print(f"User Followers Count: {user.data.public_metrics['followers_count']}")
                    print("-" * 50)

                if 'next_token' in response.meta:
                    nextPage = response.meta['next_token']
                else:
                    return tweets
                
            except Exception as e:
                print(f"Error fetching tweets: {e}")
                return None
    

tweet_fields = ['created_at', 'id', 'author_id', 'text', 'public_metrics', 'geo', 'lang', 'attachments', 'entities']
user_fields = ['id', 'name', 'username', 'description', 'created_at', 'public_metrics']

twitter = TWITTER("TWITTER_TOKEN")
tweets =  twitter.getTweets("formula 1", tweet_fields, user_fields)

print(tweets)


