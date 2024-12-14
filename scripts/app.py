import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

# Display title and description
st.title('Financial News Sentiment Analysis and Stock Data')
st.write("""
This web app analyzes financial news and stock price trends using sentiment analysis and exploratory data analysis.
""")

# Load news data
news_data_path = os.path.abspath('./data/news_data.csv')
st.write(f"News Data Path: {news_data_path}")
news_df = pd.read_csv(news_data_path)

# Load stock data
stock_files = [
    'AAPL_historical_data.csv', 'AMZN_historical_data.csv',
    'META_historical_data.csv', 'MSFT_historical_data.csv',
    'NVDA_historical_data.csv', 'TSLA_historical_data.csv'
]
stock_df_list = []
for file in stock_files:
    df = pd.read_csv(os.path.join('./data/yfinance_data', file))
    df['stock_symbol'] = file.split('_')[0].upper()  # Extract stock symbol from file name
    stock_df_list.append(df)
stock_df = pd.concat(stock_df_list, ignore_index=True)

# Show basic data information
st.subheader('News Data Info')
st.write(news_df.info())
st.subheader('Stock Data Info')
st.write(stock_df.info())

# Display first few rows of news and stock data
st.subheader('News Data Head')
st.write(news_df.head())

st.subheader('Stock Data Head')
st.write(stock_df.head())

# Sentiment Analysis
st.subheader('Sentiment Analysis of Headlines')
if 'headline' in news_df.columns:
    def calculate_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)

    news_df['sentiment'] = news_df['headline'].apply(calculate_sentiment)
    st.write("Sentiment scores added to news data.")

    # Sentiment Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(news_df['sentiment'], kde=True, bins=20, color='skyblue', ax=ax)
    ax.set_title('Sentiment Score Distribution')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# Descriptive Statistics
st.subheader('Descriptive Statistics')
if 'headline' in news_df.columns:
    news_df['headline_length'] = news_df['headline'].str.len()
    st.write(news_df[['headline', 'headline_length']].describe())
    st.write("Average Headline Length:", news_df['headline_length'].mean())

    # Count articles per publisher
    publisher_counts = news_df['publisher'].value_counts()
    st.write("Top Publishers:")
    st.bar_chart(publisher_counts.head(10))

    # Analyze publication dates
    if 'date' in news_df.columns:
        try:
            # Attempt to parse the date with a flexible approach
            news_df['date'] = pd.to_datetime(news_df['date'], format='mixed', utc=True, errors='coerce')
            st.write("Date parsing successful. Example dates:")
            st.write(news_df['date'].head())
        except Exception as e:
            st.error(f"Error parsing dates: {e}")

        # Handle missing or invalid dates
        if news_df['date'].isnull().any():
            st.warning("Some dates could not be parsed. Replacing invalid dates with NaT.")
            news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')

        news_df['day_of_week'] = news_df['date'].dt.day_name()
        date_counts = news_df['date'].dt.date.value_counts().sort_index()
        st.line_chart(date_counts)

# Word Cloud for Headlines
st.subheader('Common Words in Headlines')
headline_text = " ".join(news_df['headline'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(headline_text)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

# Time Series Analysis
st.subheader('Publication Frequency Over Time')
if 'date' in news_df.columns:
    st.line_chart(news_df['date'].dt.date.value_counts().sort_index())

# Topic Modeling (Keywords Extraction)
st.subheader('Topic Modeling')
vectorizer = CountVectorizer(stop_words='english', max_features=20)
X = vectorizer.fit_transform(news_df['headline'].dropna())
keywords = vectorizer.get_feature_names_out()
st.write("Top Keywords in Headlines:", keywords)

# Plot stock closing price trends
st.subheader('Stock Price Trends')
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
fig, ax = plt.subplots(figsize=(14, 7))
for stock in stock_df['stock_symbol'].unique():
    stock_subset = stock_df[stock_df['stock_symbol'] == stock]
    ax.plot(stock_subset['Date'], stock_subset['Close'], label=stock)

ax.set_title('Stock Closing Prices Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')
ax.legend()
st.pyplot(fig)

# Analyze publisher activity
st.subheader('Publisher Analysis')
if 'publisher' in news_df.columns:
    news_df['email_domain'] = news_df['publisher'].str.extract(r'@([\w\.-]+)')
    domain_counts = news_df['email_domain'].value_counts()
    st.write("Top Email Domains Contributing to News:")
    st.bar_chart(domain_counts.head(10))
