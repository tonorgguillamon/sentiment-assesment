import src.models as models
from src.virtual_twitter import Tweet, getDummyTweets
from pathlib import Path
import os


def main():

    dummyTweetsFile = Path(os.getcwd()) / "resources" / "tariff_tweets_unique.json"
    raw_tweets = getDummyTweets(dummyTweetsFile)
    tweets = [Tweet(**raw_tweet) for raw_tweet in raw_tweets]

    print(tweets)

if __name__ == '__main__':
    main()