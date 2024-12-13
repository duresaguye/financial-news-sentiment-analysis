import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st

# Display title and description
st.title('Financial News Sentiment Analysis and Stock Data')
st.write("""
This web app shows sentiment analysis of financial news and stock price trends.
""")

# Load news data
news_data_path = os.path.abspath('./data/news_data.csv')
st.write(f"News Data Path: {news_data_path}")
news_df = pd.read_csv(news_data_path)

# Load stock data
stock_files = ['AAPL_historical_data.csv', 'AMZN_historical_data.csv', 'META_historical_data.csv', 'MSFT_historical_data.csv', 'NVDA_historical_data.csv', 'TSLA_historical_data.csv']
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

# Check if 'sentiment' column exists in the news data
if 'sentiment' in news_df.columns:
    st.subheader('Sentiment Distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(news_df['sentiment'], kde=True, bins=20, color='skyblue', ax=ax)
    ax.set_title('Sentiment Score Distribution')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)  # Ensure fig is passed to st.pyplot
else:
    st.write("Sentiment column not found in the news data.")

# Convert 'Date' column in stock data to datetime
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

# Plot stock closing price trends
fig, ax = plt.subplots(figsize=(14, 7))
for stock in stock_df['stock_symbol'].unique():
    stock_subset = stock_df[stock_df['stock_symbol'] == stock]
    ax.plot(stock_subset['Date'], stock_subset['Close'], label=stock)

ax.set_title('Stock Closing Prices Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')
ax.legend()

# Display the stock price plot in Streamlit
st.pyplot(fig)  # Ensure fig is passed to st.pyplot
