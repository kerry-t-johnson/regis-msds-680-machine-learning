#! /usr/bin/env python3
import logging
import tweepy
import urllib.parse
import yaml

LOGGER = logging.getLogger('tweets.py')

class TwitterApi:
    MAX_TWEETS              = 1000
    MAX_TWEETS_PER_REQUEST  = 100
    US_WOEID                = 23424977

    @staticmethod
    def from_credentials(path):
        with open(path) as secrets_file:
            secrets = yaml.safe_load(secrets_file)
            return TwitterApi(secrets['twitter'])

    def __init__(self, credentials) -> None:
        auth = tweepy.OAuthHandler(credentials['consumer_key'],
                                   credentials['consumer_key_secret'])
        auth.set_access_token(credentials['access_token'],
                              credentials['access_token_secret'])

        self._api = tweepy.API(auth)

    def us_hashtags(self):
        response    = self._api.get_place_trends(TwitterApi.US_WOEID).pop()
        hashtags    = []

        for htag in response['trends']:
            hashtags.append(htag['name'])

        return hashtags

    def hashtag_tweets(self, tag, limit=None, ignore_retweets=True):
        limit           = limit or TwitterApi.MAX_TWEETS
        hashtag_tweets  = []
        max_id          = None

        while True:
            tweets = self._api.search_tweets(urllib.parse.quote(tag),
                                             count=TwitterApi.MAX_TWEETS_PER_REQUEST,
                                             lang='en',
                                             result_type='recent',
                                             include_entities=False,
                                             max_id=max_id)

            for t in tweets:
                max_id = t.id

                if not ignore_retweets or not hasattr(t, 'retweeted_status'):
                    hashtag_tweets.append(t._json)

                    if len(hashtag_tweets) == limit:
                        return hashtag_tweets

            if len(tweets) < TwitterApi.MAX_TWEETS_PER_REQUEST:
                raise RuntimeError(f'Received fewer than expected results for "{tag}": {len(hashtag_tweets)}')

    def rate_limit_status(self):
        response    = self._api.rate_limit_status(resources='search,trends')
        return response['resources']

    def test(self):
        return self._api.home_timeline()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Query the Twitter API')
    parser.add_argument('-f',
                        '--secrets-file',
                        help='Path to YAML file which contains twitter.{consumer_key(_secret),access_token(_secret)}')
    parser.add_argument('-t',
                        '--tag',
                        help='Return Tweets with the specified hashtag')
    parser.add_argument('-s',
                        '--status',
                        action='store_true',
                        help='Return the rate limit status')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)-6s - %(name)10s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    for logger in ['tweepy.api', 'requests_oauthlib', 'oauthlib', 'urllib3']:
        logging.getLogger(logger).setLevel(logging.ERROR)

    api = TwitterApi.from_credentials(args.secrets_file)

    if args.status:
        print(api.rate_limit_status())
    else:
        tweets = api.hashtag_tweets(args.tag) if args.tag else api.test()
        for tweet in tweets:
            print(tweet)
