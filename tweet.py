import tweepy
import csv
import pandas as pd
import random
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import requests


company_name = st.text_input("enter company name:")
start_date = st.date_input("enter start date:")
end_date = st.date_input("enter end date:")

data = yf.download(company_name,
                   start=start_date,
                   end=end_date)
information = yf.Ticker(company_name)
information = information.info


consumer_key = 'BhtVo50GC7aJXVqKycP9UW0xw'
consumer_secret = 'eXszxFB2SQXXFBcQ7p971rqe5dIopfp4FIckZ2uElu3C6YO4rN'
access_token = '1516255772501745665-kqvEBudgF1qzvsSgbl2Vr2T32hxIT1'
access_token_secret = 'cZQes6LAGbMIMfq3OwWqAiLZDJHN5yDXwJaj9FQVYr11f'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

fetch_tweets=tweepy.Cursor(api.search_tweets, q="company_name",count=100, lang ="en", tweet_mode="extended").items()
data=pd.DataFrame(data=[[tweet_info.created_at.date(),tweet_info.full_text]for tweet_info in fetch_tweets],columns=['Date','Tweets'])
st.pyplot(data)
