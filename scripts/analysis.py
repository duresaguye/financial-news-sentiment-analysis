import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from textblob import TextBlob
from datetime import datetime
import numpy as np
import talib as ta
from pynance import get_data
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Financial News Sentiment Analysis')
parser.add_argument('--news_data', type=str, required=True, help='Path to the news data CSV file')
parser.add_argument('--stock_data', type=str, required=True, help='Path to the stock prices CSV file')
args = parser.parse_args()

# Load data
news_df = pd.read_csv(args.news_data)  # Columns: 'headline', 'publisher', 'date', 'url', 'stock_symbol'
stock_df = pd.read_csv(args.stock_data)  # Columns: 'stock_symbol', 'date', 'open', 'high', 'low', 'close', 'volume'

# Convert 'date' columns to datetime
news_df['date'] = pd.to_datetime(news_df['date'])
stock_df['date'] = pd.to_datetime(stock_df['date'])

# Task 1: Descriptive Statistics

# Headline Lengths
news_df['headline_length'] = news_df['headline'].apply(len)
headline_stats = news_df['headline_length'].describe()

# Articles per Publisher
publisher_counts = news_df['publisher'].value_counts()

# Publication Trends
publication_trends = news_df.groupby(news_df['date'].dt.date).size()

# Sentiment Analysis on Headlines
def sentiment_analysis(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

news_df['sentiment'] = news_df['headline'].apply(sentiment_analysis)

# Task 2: Topic Modeling
vectorizer = CountVectorizer(stop_words='english', max_features=100)
X = vectorizer.fit_transform(news_df['headline'])
wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0])))

# Task 3: Time Series Analysis
# Count articles published per day
article_count_per_day = news_df.groupby(news_df['date'].dt.date).size()

# Task 4: Publisher Analysis
publisher_analysis = news_df.groupby('publisher').agg(
    total_articles=('headline', 'count'),
    avg_sentiment=('sentiment', 'mean')
)

# Task 5: Stock Data Analysis with TA-Lib & PyNance

# Merge news and stock data on 'date' and 'stock_symbol'
merged_df = pd.merge(news_df, stock_df, on=['stock_symbol', 'date'], how='inner')

# Calculate Moving Averages, RSI, and MACD with TA-Lib
merged_df['SMA_50'] = ta.SMA(merged_df['close'], timeperiod=50)
merged_df['SMA_200'] = ta.SMA(merged_df['close'], timeperiod=200)
merged_df['RSI'] = ta.RSI(merged_df['close'], timeperiod=14)
merged_df['MACD'], merged_df['MACD_signal'], merged_df['MACD_hist'] = ta.MACD(merged_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Task 6: Correlation Analysis between Sentiment and Stock Returns
# Calculate daily stock returns
merged_df['daily_return'] = merged_df['close'].pct_change()

# Correlation between sentiment and stock return
correlation = merged_df[['sentiment', 'daily_return']].corr().iloc[0, 1]

# Visualizations
plt.figure(figsize=(12, 6))

# Sentiment Distribution
#  label='200-day SMA', linestyle='--')
plt.title('Stock Price and Moving Averages')
plt.legend()

plt.tight_layout()
plt.show()

# Output Results
print(f"Headline Stats:\n{headline_stats}")
print(f"Publisher Counts:\n{publisher_counts.head()}")
print(f"Sentiment Analysis:\n{news_df[['headline', 'sentiment']].head()}")
print(f"Topic Modeling Word Cloud:")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
print(f"Correlation between Sentiment and Stock Returns: {correlation}")
