import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def add_technical_indicators(df):
    """
    Adds technical indicators to the dataframe.
    Expects columns: 'Open', 'High', 'Low', 'Close', 'Volume'
    """
    df = df.copy()
    
    # Handle multi-index columns if present (yfinance often returns multi-index)
    # yfinance output typically has Ticker as level 1 if multiple tickers, 
    # but here we process one dataframe at a time.
    # If the columns are ('Adj Close', 'AAPL'), we might need to flatten or select.
    # Let's assume the input is a single ticker dataframe.
    # However, yfinance 0.2+ returns (Price, Ticker) multiindex even for single ticker if not careful,
    # or just Price as columns.
    
    # Check if columns are MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # If it's (Price, Ticker), we usually want just Price.
        # But if we downloaded multiple tickers, we should pass one ticker's df here.
        # Let's assume standard columns.
        # If the level 1 is ticker, we drop it.
        try:
           df.columns = df.columns.droplevel(1)
        except:
           pass

    # Basic check
    required_cols = ['Close']
    for col in required_cols:
        if col not in df.columns:
            # Try finding it case insensitive
            found = False
            for c in df.columns:
                if c.lower() == col.lower():
                    df.rename(columns={c: col}, inplace=True)
                    found = True
                    break
            if not found:
                 # If still not found, maybe it's 'Adj Close'
                 if 'Adj Close' in df.columns:
                     df['Close'] = df['Adj Close']
                 else:
                     raise ValueError(f"Dataframe must contain '{col}' column. Available: {df.columns}")

    # Fill missing values
    df = df.ffill().bfill()

    # SMA
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    
    # RSI
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    
    # Simple Returns
    df['Returns'] = df['Close'].pct_change()
    
    # Log Returns
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Drop NaN created by indicators
    df.dropna(inplace=True)
    
    return df

def add_sentiment_score(df):
    """
    Adds a synthetic sentiment score.
    Range: -1 to 1
    """
    # Placeholder: Random walk sentiment
    np.random.seed(42)
    sentiment = np.random.normal(0, 0.1, size=len(df))
    sentiment = np.cumsum(sentiment)
    # Normalize to -1 to 1
    sentiment = np.tanh(sentiment)
    
    df['Sentiment'] = sentiment
    return df
