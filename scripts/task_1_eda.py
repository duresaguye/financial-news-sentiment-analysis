import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

# Create directories for saving results
os.makedirs('./results', exist_ok=True)

# Print current working directory
print("Current Working Directory:", os.getcwd())

# Load data
news_data_path = os.path.abspath('./data/news_data.csv')
news_df = pd.read_csv(news_data_path)

# Load stock data from individual files
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

# Sentiment Analysis
def calculate_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)

if 'headline' in news_df.columns:
    news_df['sentiment'] = news_df['headline'].apply(calculate_sentiment)
    print("Sentiment scores added to news data.")

# Plot the distribution of sentiment scores
if 'sentiment' in news_df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(news_df['sentiment'], kde=True, bins=20, color='skyblue')
    plt.title('Sentiment Score Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.savefig('./results/news_sentiment_distribution.png')
    plt.show()

# Descriptive Statistics
if 'headline' in news_df.columns:
    news_df['headline_length'] = news_df['headline'].str.len()
    print("Descriptive Statistics for Headline Length:")
    print(news_df['headline_length'].describe())

    # Word Cloud for Headlines
    headline_text = " ".join(news_df['headline'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(headline_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of News Headlines')
    plt.savefig('./results/news_headline_wordcloud.png')
    plt.show()

# Topic Modeling (Keywords Extraction)
vectorizer = CountVectorizer(stop_words='english', max_features=20)
if 'headline' in news_df.columns:
    X = vectorizer.fit_transform(news_df['headline'].dropna())
    keywords = vectorizer.get_feature_names_out()
    print("Top Keywords in Headlines:", keywords)

# Plot stock closing price trends
plt.figure(figsize=(14, 7))
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
for stock in stock_df['stock_symbol'].unique():
    stock_subset = stock_df[stock_df['stock_symbol'] == stock]
    plt.plot(stock_subset['Date'], stock_subset['Close'], label=stock)

plt.title('Stock Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.savefig('./results/stock_prices_over_time.png')
plt.show()

# Publication Frequency Over Time
if 'date' in news_df.columns:
    news_df['date'] = pd.to_datetime(news_df['date'], utc=True)
    publication_counts = news_df['date'].dt.date.value_counts().sort_index()
    plt.figure(figsize=(14, 7))
    plt.plot(publication_counts.index, publication_counts.values, marker='o')
    plt.title('Publication Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.savefig('./results/publication_frequency.png')
    plt.show()
