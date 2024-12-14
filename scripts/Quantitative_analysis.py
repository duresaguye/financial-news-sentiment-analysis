import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
import talib as ta

# Display title and description
st.title('Stock Data Analysis with Technical Indicators')
st.write("""
This web app analyzes stock price trends using basic technical indicators (SMA, RSI, MACD) and visualizes the results.
""")

# Load stock data from multiple CSV files
stock_files = [
    'AAPL_historical_data.csv', 'AMZN_historical_data.csv',
    'META_historical_data.csv', 'MSFT_historical_data.csv',
    'NVDA_historical_data.csv', 'TSLA_historical_data.csv',
    'GOOG_historical_data.csv'
]

stock_df_list = []

# Read each CSV file and extract stock symbol
for file in stock_files:
    file_path = os.path.join('./data/yfinance_data', file)  # Adjust path as necessary
    df = pd.read_csv(file_path)
    df['stock_symbol'] = file.split('_')[0].upper()  # Extract stock symbol from file name
    stock_df_list.append(df)

# Combine all stock data into a single DataFrame
stock_df = pd.concat(stock_df_list, ignore_index=True)

# Show basic data information
st.subheader('Combined Stock Data Info')
st.write(stock_df.info())

# Display first few rows of stock data
st.subheader('Stock Data Head')
st.write(stock_df.head())

# Apply TA-Lib Indicators (Technical Analysis on Stock Data)
st.subheader('Technical Indicators (TA-Lib) on Stock Data')

# Calculate Simple Moving Average (SMA) for each stock
stock_df['SMA_50'] = stock_df.groupby('stock_symbol')['Close'].transform(lambda x: ta.SMA(x.to_numpy(), timeperiod=50))
stock_df['SMA_200'] = stock_df.groupby('stock_symbol')['Close'].transform(lambda x: ta.SMA(x.to_numpy(), timeperiod=200))

# Calculate Relative Strength Index (RSI) for each stock
stock_df['RSI'] = stock_df.groupby('stock_symbol')['Close'].transform(lambda x: ta.RSI(x.to_numpy(), timeperiod=14))

# Calculate MACD (Moving Average Convergence Divergence) for each stock
macd_df = stock_df.groupby('stock_symbol')['Close'].apply(
    lambda x: pd.DataFrame(ta.MACD(x.to_numpy(), fastperiod=12, slowperiod=26, signalperiod=9)).T
).reset_index(level=0, drop=True).reset_index(drop=True)
stock_df = pd.concat([stock_df, macd_df], axis=1)
stock_df.columns = list(stock_df.columns[:-3]) + ['MACD', 'MACD_signal', 'MACD_hist']

# Bollinger Bands Calculation for each stock
bb_df = stock_df.groupby('stock_symbol')['Close'].apply(
    lambda x: pd.DataFrame(ta.BBANDS(x.to_numpy(), timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)).T
).reset_index(level=0, drop=True).reset_index(drop=True)
stock_df = pd.concat([stock_df, bb_df], axis=1)
stock_df.columns = list(stock_df.columns[:-3]) + ['upper_band', 'middle_band', 'lower_band']

# Plot the stock closing price along with SMA, RSI, MACD, and Bollinger Bands indicators

# Plot Closing Price, SMA50, and SMA200 for each stock
for symbol in stock_df['stock_symbol'].unique():
    fig, ax = plt.subplots(figsize=(14, 7))
    symbol_df = stock_df[stock_df['stock_symbol'] == symbol]
    
    ax.plot(symbol_df['Date'], symbol_df['Close'], label='Close Price', color='blue')
    ax.plot(symbol_df['Date'], symbol_df['SMA_50'], label='50-Day SMA', color='orange')
    ax.plot(symbol_df['Date'], symbol_df['SMA_200'], label='200-Day SMA', color='green')
    
    ax.set_title(f'{symbol} Stock Closing Prices with SMA Indicators')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

# Plot RSI Indicator for each stock
for symbol in stock_df['stock_symbol'].unique():
    fig, ax = plt.subplots(figsize=(14, 7))
    symbol_df = stock_df[stock_df['stock_symbol'] == symbol]
    
    ax.plot(symbol_df['Date'], symbol_df['RSI'], label='RSI', color='red')
    ax.axhline(70, color='gray', linestyle='--', label='Overbought (70)')
    ax.axhline(30, color='gray', linestyle='--', label='Oversold (30)')
    
    ax.set_title(f'{symbol} RSI Indicator')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI Value')
    ax.legend()
    st.pyplot(fig)

# Plot MACD Indicator for each stock
for symbol in stock_df['stock_symbol'].unique():
    fig, ax = plt.subplots(figsize=(14, 7))
    symbol_df = stock_df[stock_df['stock_symbol'] == symbol]
    
    ax.plot(symbol_df['Date'], symbol_df['MACD'], label='MACD', color='blue')
    ax.plot(symbol_df['Date'], symbol_df['MACD_signal'], label='MACD Signal', color='orange')
    ax.bar(symbol_df['Date'], symbol_df['MACD_hist'], label='MACD Histogram', color='gray', alpha=0.3)
    
    ax.set_title(f'{symbol} MACD Indicator')
    ax.set_xlabel('Date')
    ax.set_ylabel('MACD Value')
    ax.legend()
    st.pyplot(fig)

# Plot Bollinger Bands for each stock
for symbol in stock_df['stock_symbol'].unique():
    fig, ax = plt.subplots(figsize=(14, 7))
    symbol_df = stock_df[stock_df['stock_symbol'] == symbol]
    
    ax.plot(symbol_df['Date'], symbol_df['Close'], label='Close Price', color='blue')
    ax.plot(symbol_df['Date'], symbol_df['upper_band'], label='Upper Band', color='green')
    ax.plot(symbol_df['Date'], symbol_df['middle_band'], label='Middle Band', color='orange')
    ax.plot(symbol_df['Date'], symbol_df['lower_band'], label='Lower Band', color='red')
    ax.fill_between(symbol_df['Date'], symbol_df['upper_band'], symbol_df['lower_band'], color='gray', alpha=0.2)
    
    ax.set_title(f'{symbol} Bollinger Bands Indicator')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

# End of the analysis
st.write("End of analysis. Thank you!")
