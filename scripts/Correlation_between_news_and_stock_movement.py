import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import streamlit as st

# Downloading NLTK data files (only need to run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load the news data
df_news = pd.read_csv('./data/news_data.csv')
df_news.columns = df_news.columns.str.strip()  # Remove any leading/trailing whitespace from column names

# Convert 'date' to datetime format and remove timezone
df_news['date'] = pd.to_datetime(df_news['date'], errors='coerce').dt.tz_localize(None)

# Initialize the PorterStemmer
ps = PorterStemmer()

# Define a function to clean and process the text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Perform stemming
    words = [ps.stem(word) for word in words]
    
    # Rejoin the words into a single string
    processed_text = ' '.join(words)
    return processed_text

# Define a function to calculate sentiment using TextBlob
def calculate_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply text preprocessing and sentiment analysis to the DataFrame
df_news['cleaned_headline'] = df_news['headline'].apply(preprocess_text)
df_news['sentiment'] = df_news['cleaned_headline'].apply(calculate_sentiment)

# Sentiment categorization (Negative, Neutral, Positive)
df_news['sentiment_category'] = pd.cut(df_news['sentiment'], bins=[-1, -0.01, 0.01, 1], labels=['Negative', 'Neutral', 'Positive'])

# Plot the Sentiment Categories
plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment_category', data=df_news, palette='viridis', hue='sentiment_category', dodge=False)
plt.title('Count of Headlines by Sentiment Category')
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
st.pyplot(plt)

# Histogram of sentiment scores
plt.figure(figsize=(10, 6))
sns.histplot(df_news['sentiment'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
st.pyplot(plt)

# 1. Load Stock Data
stock_files = [
    'AAPL_historical_data.csv', 'AMZN_historical_data.csv',
    'META_historical_data.csv', 'MSFT_historical_data.csv',
    'NVDA_historical_data.csv', 'TSLA_historical_data.csv',
    'GOOG_historical_data.csv'
]

stock_df_list = []
for file in stock_files:
    file_path = os.path.join('./data/yfinance_data', file)  
    df = pd.read_csv(file_path)
    df['stock'] = file.split('_')[0].upper()  # Extract stock symbol from filename
    stock_df_list.append(df)

# Concatenate all stock data into a single DataFramex
stock_df = pd.concat(stock_df_list, ignore_index=True)

# Process stock data
stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)

# Calculate daily returns for each company
stock_df['daily_returns'] = stock_df.groupby('stock')['Close'].pct_change()

# Rename 'Date' to 'date' for consistency with news data
stock_df = stock_df.rename(columns={'Date': 'date'})

# Merge stock and sentiment data on the 'date' column
merged_df = pd.merge(stock_df, df_news[['date', 'sentiment', 'stock']], on=['date', 'stock'], how='inner')

# Plot daily stock returns by company
plt.figure(figsize=(12, 6))
sns.lineplot(data=merged_df, x='date', y='daily_returns', hue='stock', marker="o")
plt.title('Daily Stock Returns for Companies', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Returns (%)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)

# Correlation matrix heatmap
correlation_matrix = merged_df[['daily_returns', 'sentiment']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Correlation Matrix of Daily Returns and Sentiment')
st.pyplot(plt)

# Create a visualization for the correlation between sentiment and daily returns
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df, x='sentiment', y='daily_returns', hue='stock', palette='tab10')
plt.title('Sentiment vs Daily Stock Returns')
plt.xlabel('Sentiment Score')
plt.ylabel('Daily Stock Return')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.grid(True)
plt.tight_layout()
st.pyplot(plt)
