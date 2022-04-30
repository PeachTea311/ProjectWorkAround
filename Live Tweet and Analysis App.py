import requests
import os
import json
import pandas as pd
import csv
from datetime import datetime, timedelta
import dateutil.parser
import unicodedata
import time
import snscrape.modules.twitter as sntwitter
from textblob import TextBlob
import nltk
import matplotlib.pyplot as plt 
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from tqdm import tqdm

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def filter_punc(text):
    punc = '"$%&\'()*+-/:<=>?@[\\]^_`{|}~'
    temp = ''.join([c for c in text if ord(c)<128])
    return temp.translate(str.maketrans('', '', punc))


def run_processing_steps(tweets_df):
    
    tweets_df.drop_duplicates(inplace=True)
    # 先drop
    tweets_df.dropna(subset=['content'], inplace=True)
    
    tmp = tweets_df.content.str.findall("#\w+")
    # print(tmp[tmp.apply(lambda x: x!=[])]) # 查看存在#的行

    tmp = tweets_df.content.str.findall("@\w+")
    # print(tmp[tmp.apply(lambda x: x!=[])]) # 查看存在@的行
    
    # 移除content的@ 和 \n
    tweets_df.content = tweets_df.content.str.replace("\n", "")
    # 去除https
    tweets_df.content = tweets_df.content.str.replace(r"https*\S+", "", regex=True)
    # &符号 -> and
    tweets_df.content = tweets_df.content.str.replace("&amp;", "and")
    tweets_df.content = tweets_df.content.str.replace(r"@\w+", "", regex=True)
    # 1提取tag
    tmp = tweets_df.content.str.findall("#\w+")
    # print(tmp[tmp.apply(lambda x: x!=[])]) # 查看存在#的行

    # 新建列接受#的内容，列`tags`
    tweets_df['tags'] = tmp # 存为list，新的一列
    # print(tmp)

    # 2 并移除# 【但保留word】 因为【很多#，是内容的一部分】
    # tweets_df.content = tweets_df.content.str.replace(r"#\w+", "", regex=True)
    tweets_df.content = tweets_df.content.str.replace(r"#", " ", regex=True)
    
    tweets_df.content = tweets_df.content.apply(lambda x: deEmojify(x))
    tweets_df.content = tweets_df.content.str.replace(r'\s{2,}', " ", regex=True)
    tweets_df.content = tweets_df.content.apply(filter_punc)
    
    # 处理完所有的 再drop nan
    tweets_df.dropna(subset=['content'], inplace=True)
    return tweets_df

def get_tweets(keyword, start_time, end_time, limit):
    tweets_list = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} since_time:{start_time} until_time:{end_time} lang:en').get_items()):
        if i > limit:
            break
        tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.lang, tweet.replyCount, tweet.retweetCount, tweet.likeCount, tweet.quoteCount])
    tweets_df = pd.DataFrame(tweets_list, columns=['datetime', 'id', 'content' ,'username', 'language', 'reply_count', 'retweet_count', 'like_count', 'quote_count'])
    return tweets_df

def sentiment_analysis(df):
    sentiments_r = dict()
    Russia_sentiment = []
    for ind, row in df.iterrows():
        content = row.content
        blob = TextBlob(content)
        score = blob.sentiment.polarity
        Russia_sentiment.append(score)
        if score > 0:
            sentiments_r[content] = 1
        elif score == 0:
            sentiments_r[content] = 0
        else:
            sentiments_r[content] = -1
    label = list(sentiments_r.values())
    df_label= pd.DataFrame({'label':label})
    df = df.join(df_label)
    return df

def twitter_bot(topic, limit, duration_in_min):
    current_dt = datetime.now()
    new_dt = current_dt - timedelta(hours=0,minutes=duration_in_min)
    end_time = int(current_dt.timestamp())
    start_time = int(new_dt.timestamp())
    print("running")
    
    filename = "labeled_tweets_"+ current_dt.strftime("%Y-%m-%d %H:%M") 

    try:
        print("Getting Tweets for time between ", new_dt.strftime("%Y-%m-%d %H:%M"), ' and ',
             current_dt.strftime("%Y-%m-%d %H:%M"))
        tweets_df = get_tweets(topic, start_time, end_time, limit)
        # tweets_df_russia = get_tweets('Russia', start_time, end_time, limit)
        processed_df = run_processing_steps(tweets_df)
        df= sentiment_analysis(processed_df)
        return df
    
    except Exception as e:
        print('It is not working...')
        print(e)

#To Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """

def main():
    """ Common ML Dataset Explorer """
    #st.title("Live twitter Sentiment analysis")
    #st.subheader("Select a topic which you'd like to get the sentiment analysis on :")

    html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:9px">Live twitter Sentiment analysis</p></div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("Select a topic which you'd like to get the sentiment analysis on (Ukraine/Russia) :")


    df = pd.DataFrame(columns = ['datetime', 'id', 'content', 'username', 'language', 'reply_count',
       'retweet_count', 'like_count', 'quote_count', 'tags', 'label'])
    # Collect Input from user :
    Topic = str()
    Topic = str(st.text_input("Enter the topic you are interested in (Press Enter once done)"))     
        
    Limit = st.number_input("Enter the Max No. of Tweets you would like to see at one time")     

#     Limit = int(st.text_input("Enter the Max No. of Tweets you would like to see at one time"))     
    
    if len(Topic) > 0 and len(Limit) > 0:
        
        # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
        with st.spinner("Please wait, Tweets are being extracted"):
            df = twitter_bot(Topic, Limit, 10)
        st.success('Tweets have been Extracted !!!!')    
        
        # Write Summary of the Tweets
        st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic,len(df.Tweet)))
        st.write("Total Positive Tweets are : {}".format(len(df[df["label"]==1])))
        st.write("Total Negative Tweets are : {}".format(len(df[df["label"]==-1])))
        st.write("Total Neutral Tweets are : {}".format(len(df[df["label"]==0])))
    
        if st.button("Exit"):
            st.balloons()
            

if __name__ == '__main__':
    main()
