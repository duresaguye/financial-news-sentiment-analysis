# Task 1: Exploratory Data Analysis (EDA)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
# Print current working directory
print("Current Working Directory:", os.getcwd())

# Load data
news_data_path = os.path.abspath('./data/news_data.csv')

news_df = pd.read_csv(news_data_path)


# Load stock data from individual files
stock_files = ['AAPL_historical_data.csv', 'AMZN_historical_data.csv', 'META_historical_data.csv', 'MSFT_historical_data.csv', 'NVDA_historical_data.csv', 'TSLA_historical_data.csv']
stock_df_list = []
for file in stock_files:
    df = pd.read_csv(os.path.join('./data/yfinance_data', file))

    df['stock_symbol'] = file.split('_')[0].upper()  # Extract stock symbol from file name
    stock_df_list.append(df)
stock_df = pd.concat(stock_df_list, ignore_index=True)

# Check basic information about the datasets
print("News DataFrame Info:")
print(news_df.info())
print("\nStock DataFrame Info:")
print(stock_df.info())

# Preview the data
print("\nNews DataFrame Head:")
print(news_df.head())
print("\nStock DataFrame Head:")
print(stock_df.head())

# Plot the distribution of sentiment scores
if 'sentiment' in news_df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(news_df['sentiment'], kde=True, bins=20)
    plt.title('Sentiment Score Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.savefig('./results/news_sentiment_distribution.png')
    plt.show()

# Plot stock closing price trends
plt.figure(figsize=(14, 7))
for stock in stock_df['stock_symbol'].unique():
    stock_subset = stock_df[stock_df['stock_symbol'] == stock]
    plt.plot(stock_subset['Date'], stock_subset['Close'], label=stock)

plt.title('Stock Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.savefig('./results/stock_prices_over_time.png')
plt.show()