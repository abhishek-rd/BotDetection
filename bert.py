import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline
import pandas as pd
import json
import os

# Load user data
user = pd.read_json('../datasets/Twibot-22/user.json')

# Load tweets data
each_user_tweets = json.load(open("processed_data/twibot_id_tweet.json", 'r'))

# Initialize BERT feature extraction pipeline
feature_extract = pipeline(
    'feature-extraction',
    model='distilbert-base-uncased',
    tokenizer='distilbert-base-uncased',
    device=1,
    padding=True,
    truncation=True,
    max_length=50,
    add_special_tokens=True
)


def tweets_embedding():
    print('Running tweet embeddings generation')
    path = "./processed_data/twibot_tweets_tensor.pt"

    if not os.path.exists(path):
        tweets_list = []
        for i in tqdm(range(len(each_user_tweets))):
            user_tweets = each_user_tweets.get(str(i), [])
            if not user_tweets:
                total_each_person_tweets = torch.zeros(768)
            else:
                user_tweets = sorted(user_tweets, key=lambda x: x['timestamp'], reverse=True)[:15]
                tweet_tensors = []

                for tweet in user_tweets:
                    if tweet['text'] is None:
                        tweet_tensor = torch.zeros(768)
                    else:
                        tweet_embeddings = torch.tensor(feature_extract(tweet['text']))
                        tweet_tensor = torch.mean(tweet_embeddings.squeeze(0), dim=0)

                    tweet_tensors.append(tweet_tensor)

                total_each_person_tweets = torch.mean(torch.stack(tweet_tensors), dim=0)

            tweets_list.append(total_each_person_tweets)

        tweet_tensor = torch.stack(tweets_list)
        torch.save(tweet_tensor, path)
    else:
        tweet_tensor = torch.load(path)
    print('Finished')


tweets_embedding()
