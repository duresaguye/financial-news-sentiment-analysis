# Task 3: Correlation Between News Sentiment and Stock Movement
import pandas as pd
from textblob import TextBlob

# Load data
news_df = pd.read_csv('news_data.csv')
stock_df = pd.read_csv('stock_data.csv')

# Convert dates to datetime format
news_df['date'] = pd.to_datetime(news_df['date'])
stock_df['date'] = pd.to_datetime(stock_df['date'])

# Perform Sentiment Analysis on headlines
def sentiment_analysis(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

news_df['sentiment'] = news_df['headline'].apply(sentiment_analysis)

# Merge news and stock data on date and stock symbol
merged_df = pd.merge(news_df, stock_df, on=['stock_symbol', 'date'], how='inner')

# Calculate daily stock returns
merged_df['daily_return'] = merged_df['close'].pct_change()

# Correlation Analysis
correlation = merged_df[['sentiment', 'daily_return']].corr().iloc[0, 1]

# Output Results
print(f"Correlation between sentiment and stock returns: {correlation}")
