Financial News Sentiment Analysis

This project is focused on analyzing financial news data and predicting market sentiment. The dataset includes news articles with corresponding stock symbols and sentiment analysis. The project merges financial news with stock data to determine how news headlines impact stock market behavior.
Project Overview

This repository contains the code for a financial news sentiment analysis system. The system processes a large dataset of news headlines and stock data, applying machine learning techniques to predict sentiment trends. The dataset includes news articles from various publishers, stock data for specific companies, and a combination of both to help analyze how news impacts stock prices.
Features

    Data Processing:
        News data and stock data are preprocessed for analysis, ensuring clean and structured data.
        Stock data includes attributes like opening, closing, and adjusted prices, volume, and dividends.
        News data includes headline, URL, publisher, date, and associated stock symbols.

    Sentiment Analysis:
        Apply sentiment analysis to classify the headlines into positive, negative, or neutral categories.
        Predict the sentiment based on historical stock performance and market trends.

    Data Merging:
        Merge news data with stock data based on stock symbols and dates.
        This helps in analyzing how news events correlate with stock market movements.

Dataset Information
News Data

    Total Entries: 1,407,328
    Columns:
        Unnamed: 0 (ID)
        headline (News headline)
        url (Link to the news article)
        publisher (Name of the publisher)
        date (Date of the news article)
        stock (Stock symbol associated with the news)

Stock Data

    Total Entries: 40,408
    Columns:
        Date (Date of stock data)
        Open, High, Low, Close (Stock prices for the day)
        Adj Close (Adjusted closing price)
        Volume (Amount of stock traded)
        Dividends, Stock Splits (Stock-related events)
        stock_symbol (Stock symbol for the company)

Getting Started

To get started with the project, clone this repository:

git clone https://github.com/duresaguye/financial-news-sentiment-analysis.git


Installation

    Navigate to the project directory:

cd financial-news-sentiment-analysis

    Create a virtual environment:

python -m venv venv

    Activate the virtual environment:
        On Windows:

.\venv\Scripts\activate

On macOS/Linux:

        source venv/bin/activate

    Install the required packages:

pip install -r requirements.txt



