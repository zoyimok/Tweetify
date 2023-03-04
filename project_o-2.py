
from sqlalchemy import create_engine, types
from urllib.parse import quote
from sqlalchemy.sql import text

# scape data 
import pandas as pd
# from Scweet_master.Scweet.scweet import scrape
# import tools.tweet_analysis as tw
# import tools.preprocessing as pp
from datetime import datetime
# import time
import snscrape.modules.twitter as sntwitter
import re

# def init_question():
#     num_kw = int(input('Please input how many keywords for comparison: '))
#     kw = ''
#     for i in range(num_kw):
#         tem = input(f'Please input keyword {i+1}: ')
#         kw += tem + ','
#     kw_ls = kw.split(',')
#     pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
    
#     since = input('Please input the start date of tweets, e.g. YYYY-MM-DD: ')
#     until = input('Please input the end date of tweets, e.g. YYYY-MM-DD: ')
#     return kw_ls, num_kw, since, until


# def query(text, since, until):
#     q = text  # keyword
#     q += f" until:{until}"
#     q += f" since:{since}"
#     # q += ' geocode:41.4925374,-99.9018131,1500km'
#     return q


# def snscraperper(text, since, until, interval=3): #perper
#     d = interval
#     tweet_list = []

#     # create date list with specific interval s
#     dt_rng = pd.date_range(start=since, end=until, freq=f'{d}D')

#     # Scrape for each day
#     for dt in dt_rng:
#         # since to until = since + 1 day
#         q = query(text, since=datetime.strftime(dt, '%Y-%m-%d'),
#                   until=datetime.strftime(dt+pd.to_timedelta(1, 'D'), '%Y-%m-%d'))
#         print('start scraping {date}'.format(
#             date=datetime.strftime(dt, '%Y-%m-%d')))

#         counter = 0
#         try:
#             for i, tweet in enumerate(sntwitter.TwitterSearchScraper(q).get_items()):
#                 tweet_list.append([tweet.date, tweet.user.username, tweet.rawContent,  tweet.likeCount,
#                                   tweet.replyCount, tweet.retweetCount, tweet.quoteCount, tweet.url])
#                 counter += 1
#                 if counter % 200 == 0:
#                     print(f'{counter} scrapped')

#             print('finished scraping {date}, # of tweets: {no_tweet}'.format(
#                 date=datetime.strftime(dt, '%Y-%m-%d'), no_tweet=counter))
#         except:
#             print('error occured in {date}'.format(
#                 date=datetime.strftime(dt, '%Y-%m-%d')))
#             continue

#     # Creating a dataframe from the tweets list above
#     tweets_df = pd.DataFrame(tweet_list, columns=[
#                              'Timestamp', 'Username', 'Embedded_text', 'Likes', 'Comments', 'Retweets', 'Quotes', 'Tweet URL'])
#     return tweets_df

# kw_ls, num_kw, since, until = init_question()


# for i in range(num_kw):
#     tweets_df = snscraperper(kw_ls[i], since, until)
#     filename = f'df_sns{i+1}.csv'
#     tweets_df.to_csv(filename)

#---run

kw_ls=['disney california','unistudios']
#data cleaning 

dfs = [pd.read_csv(f"df_sns{i+1}.csv") for i in range(len(kw_ls))]

def change_data_type(df):
    try:
        df.columns = df.columns.str.lower()
        data_types = {'timestamp': 'datetime64[ns]', 'username': 'object', 
                      'embedded_text': 'object', 'likes': 'int32', 
                      'comments': 'int32', 'retweets': 'int32'}
        df = df.astype(data_types)
        return df
    except Exception as e:
        print(f'An error occurred while converting the data types: {e}')
        return None

for i, df in enumerate(dfs):
    df = df.drop(columns=['Unnamed: 0','Quotes', 'Tweet URL'], axis=1)
    change_data_type(df)
    df.to_csv(f"{kw_ls[i].lower().replace(' ', '_')}.csv", index=False)


tweets_df = []
for kw in kw_ls:
    filename = f"{kw.lower().replace(' ', '_')}.csv"
    tweets = pd.read_csv(filename)
    tweets['brands'] = kw
    tweets_df.append(tweets)

tweets = pd.concat(tweets_df)
# process the combined tweets DataFrame here

tweets.to_csv('brands_file.csv', index=False)



import os
import psycopg2

#create a connection to the database
password = ('1234')
engine = create_engine(f'postgresql://postgres:{password}@localhost:5432/twitter')
tweets.to_sql('tweets', engine, if_exists='replace')


from textblob import TextBlob

# Retrieve the embedded_text and brands columns from the database
query = "select embedded_text, timestamp, brands, likes, retweets from tweets"
connection = engine.connect()
tweets = pd.read_sql(text(query), con=connection)


tweets['embedded_text'] = tweets['embedded_text'].fillna('')

def textblob_polarity(text):
    text = str(text)
    return TextBlob(text).sentiment.polarity


def textblob_subjectivity(text):
    text = str(text)
    return TextBlob(text).sentiment.subjectivity


tweets['polarity'] = tweets['embedded_text'].apply(textblob_polarity)
tweets['subjectivity'] = tweets['embedded_text'].apply(textblob_subjectivity)


# print the polarity values
print(tweets['polarity'])
print(tweets['subjectivity'])
print(type(tweets))
# print(tweets.keys())


def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# Add the result column to the DataFrame
tweets['result'] = tweets['polarity'].apply(getAnalysis)
print(tweets)
tweets['result'].value_counts()

# Save the updated DataFrame to a csv file if desired
tweets.to_csv('result5_tweets.csv', index=False)

tweets = pd.read_csv('result5_tweets.csv')

positive_tweets = print(tweets['result']=='positive')
positive_tweets

a = 0
positive_count=0
negative_count=0
neutral_count=0


for i in tweets['embedded_text']:
    a = a + textblob_polarity(i)
    if textblob_polarity(i) > 0:
        positive_count += 1

    elif textblob_polarity(i) < 0:
        negative_count += 1
    else:
         neutral_count += 1


         text_count = positive_count + negative_count + neutral_count
print(text_count)

p_positive = (positive_count/text_count)*100
print(p_positive)

p_negative = (negative_count/text_count)*100
print(p_negative)

p_neutral = (neutral_count/text_count)*100
print(p_neutral)

print(tweets.columns)


##positiive tweets for NLTK
# Load the tweets into a pandas DataFrame
tweets = pd.read_csv('result5_tweets.csv')

# Get only positive tweets and sort by polarity in descending order
positive_tweets = tweets[tweets['result'] == 'Positive'].sort_values(by=['polarity'], ascending=False)

# Print the content of each positive tweet
i = 1
for index, row in positive_tweets.iterrows():
    print(f"{i}. {row['embedded_text']}")
    i += 1

# Get only negative tweets and sort by polarity in ascending order
negative_tweets = tweets[tweets['result'] == 'Negative'].sort_values(by=['polarity'], ascending=False)

# Print the content of each negative tweet
j = 1
for index, row in negative_tweets.iterrows():
    print(f"{j}. {row['embedded_text']}")
    j += 1

#----

# need to seperate by brands
#nltk
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd


## Load the tweets into a pandas DataFrame
tweets = pd.read_csv("result5_tweets.csv")
nltk.download('punkt')
nltk.download('stopwords')


# Remove stop words
stop_words = set(stopwords.words("english"))
stop_words.update(['disneyland', 'universialstudios', 'disney', 'reply', 'like','https','one','get'])

# Remove special characters, symbols, and numbers
def extract_words(text):
    words = word_tokenize(text)
    filtered_words = []
    for word in words:
        word = re.sub(r'[^a-zA-Z]', '', word)
        if word.isalpha():
            filtered_words.append(word.lower())
    return filtered_words

# Get only negative tweets and sort by frequency in descending order
negative_tweets = tweets[tweets['result'] == 'Negative'].sort_values(by=['polarity'], ascending=False)

# Extract the negative words from each negative tweet
negative_words = []
for i in negative_tweets['embedded_text']:
    for word in extract_words(i):
        if word not in stop_words:
            negative_words.append(word)

# Get the frequency of each negative word
negative_list = nltk.FreqDist(negative_words)

# Print the most common negative words
print("Most common words in negative tweets:")
print(negative_list.most_common(30))

#-----
#---nltk per tweet
# import re
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import pandas as pd

# # Load the tweets into a pandas DataFrame
# tweets = pd.read_csv("result5_tweets.csv")
# nltk.download('punkt')
# nltk.download('stopwords')

# # Remove stop words
# stop_words = set(stopwords.words("english"))
# stop_words.update(['disneyland', 'universialstudios', 'disney', 'reply', 'like'])

# # Remove special characters, symbols, and numbers
# def extract_words(text):
#     words = word_tokenize(text)
#     filtered_words = []
#     for word in words:
#         word = re.sub(r'[^a-zA-Z]', '', word)
#         if word.isalpha():
#             filtered_words.append(word.lower())
#     return filtered_words

# # Get only negative tweets
# negative_tweets = tweets[tweets['result'] == 'Negative']

# # For each negative tweet, find the most common words
# for index, row in negative_tweets.iterrows():
#     negative_words = []
#     for word in extract_words(row['embedded_text']):
#         if word not in stop_words:
#             negative_words.append(word)
#     negative_list = nltk.FreqDist(negative_words)
    
#     print("Most common words in each negative tweet:")
#     print(negative_list.most_common(30))
#     print("\n")


#-----


# need to seperate by brands
#nltk
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd


## Load the tweets into a pandas DataFrame
tweets = pd.read_csv("result5_tweets.csv")
nltk.download('punkt')
nltk.download('stopwords')


# Remove stop words
stop_words = set(stopwords.words("english"))
stop_words.update(['disneyland', 'universialstudios', 'disney', 'reply', 'like'])

# Remove special characters, symbols, and numbers
def extract_words(text):
    words = word_tokenize(text)
    filtered_words = []
    for word in words:
        word = re.sub(r'[^a-zA-Z]', '', word)
        if word.isalpha():
            filtered_words.append(word.lower())
    return filtered_words

# Print the most common negative words
negative_tweets = tweets[tweets['result'] == 'Negative']
negative_words = []
for i in negative_tweets['embedded_text']:
    for word in extract_words(i):
        if word not in stop_words:
            negative_words.append(word)
negative_list = nltk.FreqDist(negative_words)


print("Most common words in negative tweets:")
print(negative_list.most_common(30))

# Print the most common positive words
positive_tweets = tweets[tweets['result'] == 'Positive']
positive_words = []
for i in positive_tweets['embedded_text']:
    for word in extract_words(i):
        if word not in stop_words:
            positive_words.append(word)
positive_list = nltk.FreqDist(positive_words)

print("Most common words in positive tweets:")
print(positive_list.most_common(30))


# #--wordcloud
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# df = pd.read_csv("result5_tweets.csv")

# # Combine all the tweets into a single string
# all_words = ' '.join(df['embedded_text'])

# # Generate the word cloud
# cloud = WordCloud(width=500, height=300, random_state=0, max_font_size=100).generate(all_words)

# # Plot the word cloud
# plt.imshow(cloud)
# plt.show()


# #====Scattor plot

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import random

def remove_outliers(data):
    z = np.abs(stats.zscore(data))
    threshold = 1
    return data[(z < threshold)]

# Load the data from the csv file
df = pd.read_csv('result5_tweets.csv')
print(df.head(20))
# Convert the timestamp column to datetime format and extract the date
df['timestamp_2'] = pd.to_datetime(df['timestamp'])
df['datestamp'] = df['timestamp_2'].dt.date

# Filter data for the year 2022 only
df = df[(df['timestamp_2'] >= '2022-01-01') & (df['timestamp_2'] < '2023-02-11')]

# Group the data by date and brand, and count the sum of each date per brand
df_count = df.groupby(['datestamp', 'brands'])['likes', 'retweets'].sum().reset_index()

# Mapping of keywords to colors for plotting
color_mapping = {kw: f'C{i}' for i, kw in enumerate(kw_ls)}

fig, ax = plt.subplots(figsize=(20, 10))

# Loop through each keyword in the list
for kw in kw_ls:
    # Extract the data for each brand
    data = df_count[df_count['brands'] == kw].set_index('datestamp')

    # Remove outliers
    data['likes'] = remove_outliers(data['likes'])
    data['retweets'] = remove_outliers(data['retweets'])

    # Plot the scatter plot for likes and retweets for each brand
    ax.scatter(data.index, data['likes'], c=color_mapping[kw], label=kw.capitalize() + ' Likes')
    ax.scatter(data.index, data['retweets'], c=color_mapping[kw], marker='^', label=kw.capitalize() + ' Retweets')

# Add the legend and labels
ax.legend(loc='upper left')
ax.set_xlabel('Timeframe')
ax.set_ylabel('Number of Likes and Retweets per interval month')
ax.set_title('Brand Comparison: Likes and Retweets')
plt.show()


# # ====line graph

import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the csv file
df = pd.read_csv('result5_tweets.csv')

# Convert the timestamp column to datetime format and extract the date
df['timestamp_2'] = pd.to_datetime(df['timestamp'])
df['datestamp'] = df['timestamp_2'].dt.date

# Filter data for the year 2022 only
df = df[(df['timestamp_2'] >= '2022-01-01') & (df['timestamp_2'] < '2023-02-11')]


df_count = df[df['brands'].isin(kw_ls)].groupby(['datestamp', 'brands'])['timestamp_2'].count().reset_index()

# Plot the line graph with 3 lines for each brand
fig, ax = plt.subplots(figsize=(20, 10))

for brand in kw_ls:
    data = df_count[df_count['brands'] == brand].set_index('datestamp')
    data.plot(kind='line', ax=ax, legend=False)

# Add the legend and labels
ax.legend(kw_ls)
ax.set_xlabel('Timeframe')
ax.set_ylabel('Number of Tweets per interval day')
plt.show()



#====correlation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data from the csv file
df = pd.read_csv('result5_tweets.csv')

# Convert the timestamp column to datetime format and extract the date
df['timestamp_2'] = pd.to_datetime(df['timestamp'])
df['datestamp'] = df['timestamp_2'].dt.date

# Filter the data for year 2022
df = df[(df['timestamp_2'] >= '2022-01-01') & (df['timestamp_2'] <= '2023-02-11')]

# Group the data by date and brand,
df_grouped = df.groupby(['timestamp_2', 'brands']).size().reset_index(name='counts')

# Pivot the data so that brands are in columns and dates are in rows
df_pivot = df_grouped.pivot(index='timestamp_2', columns='brands', values='counts')

# Calculate the correlation between the brands
corr = df_pivot.corr()

# Plot the correlation matrix as a heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Adjust the plot size
plt.gcf().set_size_inches(10, 10)

# Show the plot
plt.show()



#====new scatter plot
import matplotlib.pyplot as plt
import pandas as pd

# Load the tweets into a pandas DataFrame
tweets = pd.read_csv('result5_tweets.csv')

# Get the polarity and subjectivity columns
polarity = tweets['polarity']
subjectivity = tweets['subjectivity']
brands = tweets['brands']

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the points with different colors based on the brand
mask_disney = brands == 'disney california'
mask_universal = brands == 'unistudios'
ax.scatter(polarity[mask_disney], subjectivity[mask_disney], c='blue', s=5, label='disney')
ax.scatter(polarity[mask_universal], subjectivity[mask_universal], c='red', s=5, label='universal')

# Add labels for the x and y axes
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')

# Add a legend
plt.legend()

# Show the plot
plt.show()



#====mola's correlation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data from the csv file
df = pd.read_csv('result5_tweets.csv')

# Convert the timestamp column to datetime format and extract the date
df['timestamp_2'] = pd.to_datetime(df['timestamp'])
df['datestamp'] = df['timestamp_2'].dt.date

# Filter the data for year 2022
df = df[(df['timestamp_2'] >= '2022-01-01') & (df['timestamp_2'] <= '2023-02-11')]

# Group the data by date and brand,
df_grouped = df.groupby(['timestamp_2', 'brands']).size().reset_index(name='counts')

# Pivot the data so that brands are in columns and dates are in rows
df_pivot = df_grouped.pivot(index='timestamp_2', columns='brands', values='counts')
df_pivotxna = df_pivot.fillna(df_pivot.mean())
# Calculate the correlation between the brands
corr = df_pivotxna.corr()
print(corr)

# Plot the correlation matrix as a heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Adjust the plot size
plt.gcf().set_size_inches(10, 10)

# Show the plot
plt.show()
