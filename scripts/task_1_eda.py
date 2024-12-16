import os
import pandas as pd
import pandas_ta as ta
import talib
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from pynance import Stock

# Load stock data (ensure it includes columns Open, High, Low, Close, Volume)
stock_files = [
    'AAPL_historical_data.csv', 'AMZN_historical_data.csv',
    'META_historical_data.csv', 'MSFT_historical_data.csv',
    'NVDA_historical_data.csv', 'TSLA_historical_data.csv',
    'GOOGL_historical_data.csv'
]

# List to store individual stock dataframes
stock_df_list = []
for file in stock_files:
    # Load stock data
    file_path = os.path.join('./data/yfinance_data', file)
    df = pd.read_csv(file_path)
    
    # Ensure the file has the required columns (Open, High, Low, Close, Volume)
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        # Add stock symbol as a new column
        df['stock_symbol'] = file.split('_')[0].upper()  # Extract stock symbol from file name
        
        # Add the dataframe to the list
        stock_df_list.append(df)
    else:
        st.write(f"Warning: Missing required columns in {file}")

# Combine all stock dataframes into a single dataframe
stock_df = pd.concat(stock_df_list, ignore_index=True)

# Display first few rows of the loaded stock data
st.write("Stock Data Overview:")
st.write(stock_df.head())

# Apply Technical Indicators (Example: SMA, RSI, MACD)
# SMA: Simple Moving Average
stock_df['SMA_50'] = ta.sma(stock_df['Close'], timeperiod=50)
stock_df['SMA_200'] = ta.sma(stock_df['Close'], timeperiod=200)

# RSI: Relative Strength Index
stock_df['RSI_14'] = ta.rsi(stock_df['Close'], timeperiod=14)

# MACD: Moving Average Convergence Divergence
macd, macd_signal, macd_hist = talib.MACD(stock_df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
stock_df['MACD'] = macd
stock_df['MACD_Signal'] = macd_signal
stock_df['MACD_Hist'] = macd_hist

# Display the stock data with calculated indicators
st.write("Stock Data with Technical Indicators (SMA, RSI, MACD):")
st.write(stock_df[['stock_symbol', 'Date', 'Close', 'SMA_50', 'SMA_200', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist']].tail())

# Visualizing the stock data and technical indicators

# Select a stock symbol for visualization (Example: AAPL)
stock_symbol = 'AAPL'  # Modify as per your need
stock_symbol_df = stock_df[stock_df['stock_symbol'] == stock_symbol]

# Plot Close price and SMAs
plt.figure(figsize=(10, 6))
plt.plot(stock_symbol_df['Date'], stock_symbol_df['Close'], label='Close Price', color='blue')
plt.plot(stock_symbol_df['Date'], stock_symbol_df['SMA_50'], label='50-day SMA', color='red', linestyle='--')
plt.plot(stock_symbol_df['Date'], stock_symbol_df['SMA_200'], label='200-day SMA', color='green', linestyle='--')

plt.title(f'{stock_symbol} Stock Price with SMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)

# Plot RSI
plt.figure(figsize=(10, 4))
plt.plot(stock_symbol_df['Date'], stock_symbol_df['RSI_14'], label='RSI 14', color='orange')
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
plt.title(f'{stock_symbol} RSI (14)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

# Display the RSI plot in Streamlit
st.pyplot(plt)

# Plot MACD
plt.figure(figsize=(10, 6))
plt.plot(stock_symbol_df['Date'], stock_symbol_df['MACD'], label='MACD', color='blue')
plt.plot(stock_symbol_df['Date'], stock_symbol_df['MACD_Signal'], label='MACD Signal', color='red', linestyle='--')
plt.bar(stock_symbol_df['Date'], stock_symbol_df['MACD_Hist'], label='MACD Histogram', color='green', alpha=0.3)
plt.title(f'{stock_symbol} MACD')
plt.xlabel('Date')
plt.ylabel('MACD Value')
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

# Display the MACD plot in Streamlit
st.pyplot(plt)

# Incorporate PyNance for Financial Metrics

# Using PyNance to calculate PE Ratio, PB Ratio, etc.
# Example: Use PyNance to fetch stock info and calculate PE and PB Ratios

stock_info = Stock(stock_symbol).info()

# Fetch PE Ratio, PB Ratio, and other metrics
pe_ratio = stock_info.get('peRatio', 'N/A')
pb_ratio = stock_info.get('priceToBook', 'N/A')
market_cap = stock_info.get('marketCap', 'N/A')

# Display these metrics in the Streamlit app
st.write(f"Financial Metrics for {stock_symbol}:")
st.write(f"PE Ratio: {pe_ratio}")
st.write(f"PB Ratio: {pb_ratio}")
st.write(f"Market Cap: {market_cap}")

# Optionally, you can calculate other metrics like Dividend Yield, Beta, etc. using PyNance.
