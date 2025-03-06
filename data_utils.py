import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import warnings

def get_api_keys():
    """Get API keys from secrets with fallbacks to default values"""
    try:
        # Try to get from secrets
        alpha_vantage_key = st.secrets["api_keys"]["alpha_vantage"]
        fred_api_key = st.secrets["api_keys"]["fred"]
    except (KeyError, FileNotFoundError):
        # Fallbacks for local development - no warning shown
        alpha_vantage_key = "6PY7F7H490LIJIXG"
        fred_api_key = "407359595d242cb6848578f701b78f83"
    
    return alpha_vantage_key, fred_api_key

def fetch_additional_stock_data(ticker):
    """Fetch additional stock data including volume, fundamentals, and options chain."""
    try:
        stock = yf.Ticker(ticker)
        
        # Get options expiration dates
        try:
            options_dates = stock.options
        except:
            options_dates = []
            
        # Get options chain for the nearest expiration date
        options_chain = None
        if options_dates and len(options_dates) > 0:
            nearest_date = options_dates[0]
            try:
                calls = stock.option_chain(nearest_date).calls
                puts = stock.option_chain(nearest_date).puts
                options_chain = {
                    'date': nearest_date,
                    'calls': calls,
                    'puts': puts
                }
            except:
                pass
        
        # Get fundamentals
        info = stock.info
        fundamentals = {
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'eps': info.get('trailingEps'),
            'dividend_yield': info.get('dividendYield'),
            'beta': info.get('beta'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            'avg_volume': info.get('averageVolume'),
            'sector': info.get('sector'),
            'industry': info.get('industry')
        }
        
        # Get recent news
        try:
            news = stock.news
        except:
            news = []
        
        # Get recommendation trends
        try:
            recommendations = stock.recommendations
        except:
            recommendations = pd.DataFrame()
        
        # Get historical data with volume
        history = stock.history(period="1y")
        
        return {
            'options_dates': options_dates,
            'options_chain': options_chain,
            'fundamentals': fundamentals,
            'news': news,
            'recommendations': recommendations,
            'history': history
        }
    except Exception as e:
        st.error(f"Error fetching additional data: {e}")
        return None

def calculate_technical_indicators(history):
    """Calculate common technical indicators from historical data."""
    df = history.copy()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Volume indicators
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    return df

def format_value(value, value_type):
    """Format values for display in metrics."""
    if value is None:
        return "N/A"
    
    if value_type == "currency":
        if value >= 1e12:
            return f"${value/1e12:.1f}T"
        elif value >= 1e9:
            return f"${value/1e9:.1f}B"
        elif value >= 1e6:
            return f"${value/1e6:.1f}M"
        else:
            return f"${value:.2f}"
    
    elif value_type == "percentage":
        return f"{value*100:.2f}%" if value else "N/A"
    
    elif value_type == "ratio":
        return f"{value:.2f}" if value else "N/A"
    
    elif value_type == "volume":
        if value >= 1e9:
            return f"{value/1e9:.1f}B"
        elif value >= 1e6:
            return f"{value/1e6:.1f}M"
        elif value >= 1e3:
            return f"{value/1e3:.1f}K"
        else:
            return f"{value:.0f}"
    
    else:  # "number"
        return f"{value:.2f}" if value else "N/A"

def generate_technical_signals(indicators):
    """Generate technical signals from indicators."""
    # Get the recent data
    recent = indicators.iloc[-10:]
    latest = indicators.iloc[-1]
    prev = indicators.iloc[-2] if len(indicators) > 1 else None
    
    signals = {
        "Summary": [],
        "Moving Averages": [],
        "Oscillators": [],
        "Trend Indicators": []
    }
    
    # Moving Averages Signals
    if latest['Close'] > latest['SMA_20']:
        signals["Moving Averages"].append({
            "direction": "bullish",
            "description": "Price above 20-day Moving Average"
        })
    else:
        signals["Moving Averages"].append({
            "direction": "bearish",
            "description": "Price below 20-day Moving Average"
        })
    
    if latest['Close'] > latest['SMA_50']:
        signals["Moving Averages"].append({
            "direction": "bullish",
            "description": "Price above 50-day Moving Average"
        })
    else:
        signals["Moving Averages"].append({
            "direction": "bearish",
            "description": "Price below 50-day Moving Average"
        })
    
    if latest['SMA_20'] > latest['SMA_50']:
        signals["Moving Averages"].append({
            "direction": "bullish",
            "description": "20-day MA above 50-day MA (Golden Cross potential)"
        })
    # Fixed this line to properly check for prev instead of using it in a boolean context
    elif latest['SMA_20'] < latest['SMA_50'] and prev is not None and prev['SMA_20'] > prev['SMA_50']:
        signals["Moving Averages"].append({
            "direction": "bearish",
            "description": "20-day MA crossed below 50-day MA (Death Cross)"
        })
    
    # Oscillator Signals
    if latest['RSI'] > 70:
        signals["Oscillators"].append({
            "direction": "bearish",
            "description": "RSI above 70 - Overbought"
        })
    elif latest['RSI'] < 30:
        signals["Oscillators"].append({
            "direction": "bullish",
            "description": "RSI below 30 - Oversold"
        })
    
    # Fixed these lines to properly check for prev
    if prev is not None and latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
        signals["Oscillators"].append({
            "direction": "bullish",
            "description": "MACD crossed above Signal line"
        })
    elif prev is not None and latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
        signals["Oscillators"].append({
            "direction": "bearish",
            "description": "MACD crossed below Signal line"
        })
    
    # Bollinger Bands Signals
    if latest['Close'] > latest['BB_Upper']:
        signals["Trend Indicators"].append({
            "direction": "bearish",
            "description": "Price above Upper Bollinger Band - Potential reversal or continuation"
        })
    elif latest['Close'] < latest['BB_Lower']:
        signals["Trend Indicators"].append({
            "direction": "bullish",
            "description": "Price below Lower Bollinger Band - Potential reversal or continuation"
        })
    
    # Trend Detection
    price_trend = "bullish" if latest['Close'] > recent['Close'].mean() else "bearish"
    signals["Trend Indicators"].append({
        "direction": price_trend,
        "description": f"Short-term price trend is {price_trend.upper()}"
    })
    
    # Summary - Count bullish vs bearish signals
    all_signals = []
    for category, category_signals in signals.items():
        if category != "Summary":
            all_signals.extend(category_signals)
    
    bullish_count = sum(1 for signal in all_signals if signal['direction'] == "bullish")
    bearish_count = sum(1 for signal in all_signals if signal['direction'] == "bearish")
    neutral_count = sum(1 for signal in all_signals if signal['direction'] == "neutral")
    
    if bullish_count > bearish_count:
        overall_sentiment = "bullish"
        sentiment_desc = f"Bullish signals ({bullish_count}) outweigh bearish signals ({bearish_count})"
    elif bearish_count > bullish_count:
        overall_sentiment = "bearish"
        sentiment_desc = f"Bearish signals ({bearish_count}) outweigh bullish signals ({bullish_count})"
    else:
        overall_sentiment = "neutral"
        sentiment_desc = f"Mixed signals: {bullish_count} bullish, {bearish_count} bearish"
    
    signals["Summary"].append({
        "direction": overall_sentiment,
        "description": sentiment_desc
    })
    
    return signals
