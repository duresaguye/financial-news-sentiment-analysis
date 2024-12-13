import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from textblob import TextBlob

# Load news data
# Print current working directory
print("Current Working Directory:", os.getcwd())

# Load news data
news_data_path = os.path.abspath('./data/news_data.csv')
print("News Data Path:", news_data_path)
news_df = pd.read_csv(news_data_path)

# Perform Sentiment Analysis
def sentiment_analysis(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

news_df['sentiment'] = news_df['headline'].apply(sentiment_analysis)

# Descriptive statistics for textual lengths (headline length)
news_df['headline_length'] = news_df['headline'].apply(len)
headline_length_stats = news_df['headline_length'].describe()

# Count the number of articles per publisher
publisher_counts = news_df['publisher'].value_counts()

# Analyze the publication dates
news_df['date'] = pd.to_datetime(news_df['date'])
publication_frequency = news_df['date'].dt.to_period('M').value_counts().sort_index()

# Sentiment Distribution Plot
plt.figure(figsize=(10, 6))
sns.histplot(news_df['sentiment'], kde=True, bins=20, color='skyblue')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.savefig('./results/news_sentiment_distribution.png')
plt.show()

# Aggregate sentiment by date
sentiment_summary = news_df.groupby('date')['sentiment'].mean().reset_index()

# Save aggregated sentiment to results
sentiment_summary.to_csv('./results/sentiment_summary.csv', index=False)

# Plot daily sentiment scores
plt.figure(figsize=(14, 7))
plt.plot(sentiment_summary['date'], sentiment_summary['sentiment'], marker='o', linestyle='-')
plt.title('Daily Average Sentiment Scores')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.grid()
plt.savefig('./results/daily_sentiment_trend.png')
plt.show()

# Load stock data from individual files
stock_files = ['AAPL_historical_data.csv', 'AMZN_historical_data.csv', 'META_historical_data.csv', 
               'MSFT_historical_data.csv', 'NVDA_historical_data.csv', 'TSLA_historical_data.csv']
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

# Convert the 'Date' column to datetime format for accurate time plotting
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

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

# Print the summary statistics and counts
print("\nHeadline Length Statistics:")
print(headline_length_stats)

print("\nPublisher Article Counts:")
print(publisher_counts)

print("\nPublication Frequency Over Time:")
print(publication_frequency)

print("\nAggregated Sentiment Summary:")
print(sentiment_summary.head())
