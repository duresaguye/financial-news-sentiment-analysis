Financial News and Stock Price Analysis
Project Overview

This project analyzes the relationship between financial news sentiment and stock market movements. Using sentiment analysis and statistical methods, it aims to uncover how the tone of financial news impacts stock prices and to develop data-driven investment strategies.
Objectives

    Sentiment Analysis: Quantify the tone of financial news headlines to understand their emotional context.
    Correlation Analysis: Identify statistical correlations between news sentiment and stock price movements.
    Predictive Strategies: Leverage sentiment insights to recommend actionable investment strategies.

Data

The dataset contains the following fields:

    headline: Title of the news article
    url: Link to the full news article
    publisher: Author/creator of the article
    date: Publication date and time (UTC-4 timezone)
    stock: Stock ticker symbol

Tasks

    
    Quantitative Analysis: Explore financial data using technical indicators and visualizations.
    Correlation Analysis: Align stock and news datasets, perform sentiment analysis, and analyze relationships between sentiment and stock returns.

Getting Started

    Clone the Repository

git clone https://github.com/duresaguye/financial-news-sentiment-analysis  
cd financial-news-sentiment-analysis  

Create a Virtual Environment
To isolate project dependencies, create a virtual environment:

python -m venv venv  

Activate the Virtual Environment

On Windows:

    venv\Scripts\activate  

On macOS/Linux:

    source venv/bin/activate  

Install Dependencies
Install the required libraries by running:

pip install -r requirements.txt  

Run the Application with Streamlit
The project contains multiple scripts for different analysis tasks. You can run each script individually with Streamlit:

    Exploratory Data Analysis (EDA)

streamlit run scripts/taks_1_eda.py  

Correlation between News Sentiment and Stock Movement

streamlit run scripts/Correlation_between_news_and_stock_movement.py  

Quantitative Analysis

        streamlit run scripts/quantitative_analysis.py  

Each of these scripts will open in your browser, and you can interact with the corresponding analysis.

