import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import re
import json
import time
from datetime import datetime, timedelta
import threading
import yfinance as yf

# Set page config
st.set_page_config(
    page_title="Meow Dashboard",
    page_icon="😺",
    layout="wide",
)

# Polygon API key
POLYGON_API_KEY = "9skphQ6G7_rESW6iTNJDIAycT9gncpje"

# Apply terminal-style CSS
terminal_css = """
<style>
    /* Main terminal theme */
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'VT323', monospace;
        color: #ffffff;
        caret-color: #4a90e2;
    }
    
    /* Background and main container */
    .main {
        background-color: #2f2f2f;
        background-image: linear-gradient(rgba(74, 144, 226, 0.05) 50%, transparent 50%);
        background-size: 100% 4px;
    }
    
    /* Windows 95-style elements */
    div.stButton > button, .stSelectbox > div > div {
        border: 2px solid #e6f3ff !important;
        border-right: 2px solid #000 !important;
        border-bottom: 2px solid #000 !important;
        background-color: #4a90e2 !important;
        color: #ffffff !important;
        font-family: 'VT323', monospace !important;
        font-size: 18px !important;
    }
    
    /* Text inputs */
    div.stTextInput > div > div > input {
        background-color: #2f2f2f;
        color: #ffffff;
        border: 1px solid #e6f3ff;
        font-family: 'VT323', monospace !important;
    }
    
    /* Metrics */
    div.stMetric > div {
        background-color: #2f2f2f;
        border: 1px solid #e6f3ff;
        padding: 10px;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e6f3ff !important;
        font-family: 'VT323', monospace !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #2f2f2f;
        border-right: 2px solid #e6f3ff;
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        background-color: #2f2f2f !important;
    }
    
    /* Windows 95-style header */
    .win95-header {
        background-color: #4a90e2;
        color: #ffffff !important;
        font-weight: bold;
        padding: 2px 5px;
        font-family: 'VT323', monospace;
        border-top: 2px solid #e6f3ff;
        border-left: 2px solid #e6f3ff;
        border-right: 2px solid #000000;
        border-bottom: 2px solid #000000;
        margin-bottom: 5px;
    }
    
    /* Windows 95-style panel */
    .win95-panel {
        background-color: #2f2f2f;
        border-top: 2px solid #e6f3ff;
        border-left: 2px solid #e6f3ff;
        border-right: 2px solid #000000;
        border-bottom: 2px solid #000000;
        padding: 5px;
        margin: 5px 0;
    }
    
    /* Dashboard cards */
    .dashboard-card {
        background-color: #2f2f2f;
        border: 1px solid #e6f3ff;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        height: 100%;
    }
    
    /* Stock price info */
    .stock-up {
        color: #4CAF50 !important;
    }
    
    .stock-down {
        color: #F44336 !important;
    }
    
    /* News item styling */
    .news-item {
        background-color: #2f2f2f;
        border: 1px solid #e6f3ff;
        padding: 8px;
        margin-bottom: 8px;
        font-family: 'VT323', monospace;
    }
    
    .news-ticker {
        font-weight: bold;
        color: #e6f3ff;
        font-size: 14px;
    }
    
    .news-headline {
        font-size: 16px;
        margin: 3px 0;
        color: #ffffff;
        text-decoration: none;
    }
    
    .news-headline:hover {
        text-decoration: underline;
        cursor: pointer;
    }
    
    .news-outcome {
        font-weight: bold;
        font-size: 14px;
    }
    
    .news-positive {
        color: #4CAF50;
    }
    
    .news-negative {
        color: #F44336;
    }
    
    .news-neutral {
        color: #FFC107;
    }
    
    /* Scrollable container */
    .scrollable {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 10px;
    }
    
    /* Refresh indicator */
    .refresh-indicator {
        position: fixed;
        top: 5px;
        right: 10px;
        color: #4a90e2;
        font-size: 12px;
        z-index: 1000;
    }
    
    /* Watchlist item */
    .watchlist-item {
        display: flex;
        justify-content: space-between;
        padding: 5px;
        border-bottom: 1px solid #4a90e2;
    }
    
    /* Badge for indicators */
    .signal-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 3px;
        font-size: 14px;
        margin: 2px;
    }
    
    .signal-buy {
        background-color: rgba(76, 175, 80, 0.3);
        color: #4CAF50;
        border: 1px solid #4CAF50;
    }
    
    .signal-sell {
        background-color: rgba(244, 67, 54, 0.3);
        color: #F44336;
        border: 1px solid #F44336;
    }
    
    .signal-hold {
        background-color: rgba(255, 193, 7, 0.3);
        color: #FFC107;
        border: 1px solid #FFC107;
    }
    
    /* Custom column widths */
    .custom-column {
        float: left;
        padding: 0 5px;
        box-sizing: border-box;
    }
    
    .col-20 {
        width: 20%;
    }
    
    .col-25 {
        width: 25%;
    }
    
    .col-33 {
        width: 33.33%;
    }
    
    .col-50 {
        width: 50%;
    }
    
    .col-66 {
        width: 66.66%;
    }
    
    .col-75 {
        width: 75%;
    }
    
    .col-80 {
        width: 80%;
    }
    
    /* Clear floats */
    .row:after {
        content: "";
        display: table;
        clear: both;
    }
    
    /* Dashboard widget */
    .dashboard-widget {
        border: 1px solid #4a90e2;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #2f2f2f;
        height: 100%;
    }
    
    /* Widget header */
    .widget-header {
        border-bottom: 1px solid #4a90e2;
        margin-bottom: 10px;
        padding-bottom: 5px;
        color: #e6f3ff;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
"""

# Function to create a Windows 95-style header
def win95_header(text):
    return f'<div class="win95-header">{text}</div>'

# Session state initialization
if 'watchlist' not in st.session_state:
    # Default watchlist
    st.session_state.watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

if 'layout' not in st.session_state:
    st.session_state.layout = {
        'watchlist': True,
        'market_overview': True,
        'stock_charts': True,
        'news': True,
        'signals': True
    }

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0

if 'stock_data' not in st.session_state:
    st.session_state.stock_data = {}

if 'market_indices' not in st.session_state:
    st.session_state.market_indices = {}

if 'news_data' not in st.session_state:
    st.session_state.news_data = []

if 'signals_data' not in st.session_state:
    st.session_state.signals_data = {}

# Functions to fetch data
def fetch_stock_data(ticker_list):
    """Fetch current stock data for watchlist"""
    result = {}
    
    try:
        # Use yfinance for quick data fetch
        tickers_data = yf.download(
            tickers=ticker_list,
            period="2d",  # Get 2 days to calculate change
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            prepost=False,
            threads=True
        )
        
        for ticker in ticker_list:
            try:
                if isinstance(tickers_data, pd.DataFrame):
                    # Single ticker case
                    current_price = tickers_data['Close'].iloc[-1]
                    prev_price = tickers_data['Close'].iloc[-2]
                else:
                    # Multiple tickers case
                    current_price = tickers_data[ticker]['Close'].iloc[-1]
                    prev_price = tickers_data[ticker]['Close'].iloc[-2]
                
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                
                result[ticker] = {
                    'price': current_price,
                    'change': change,
                    'change_pct': change_pct,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
            except Exception as e:
                result[ticker] = {
                    'price': 0.0,
                    'change': 0.0,
                    'change_pct': 0.0,
                    'error': str(e),
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
    except Exception as general_e:
        st.error(f"Error fetching stock data: {str(general_e)}")
    
    return result

def fetch_stock_intraday(ticker, interval="5m", days=1):
    """Fetch intraday data for a specific stock"""
    try:
        # Calculate period based on days
        period = f"{days}d"
        
        # Fetch data
        data = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            prepost=True
        )
        
        # Reset index to have date as a column
        data = data.reset_index()
        
        return data
    except Exception as e:
        st.error(f"Error fetching intraday data for {ticker}: {str(e)}")
        return pd.DataFrame()

def fetch_market_indices():
    """Fetch market indices data"""
    indices = {
        "^DJI": "Dow Jones",
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "^RUT": "Russell 2000",
        "^VIX": "VIX"
    }
    
    result = {}
    
    try:
        # Fetch data for all indices at once
        data = yf.download(
            tickers=list(indices.keys()),
            period="2d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True
        )
        
        for ticker, name in indices.items():
            try:
                if len(data.columns.levels) > 1:  # Multiple tickers
                    current_price = data[ticker]['Close'].iloc[-1]
                    prev_price = data[ticker]['Close'].iloc[-2]
                else:  # Single ticker
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2]
                
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                
                result[name] = {
                    'price': current_price,
                    'change': change,
                    'change_pct': change_pct,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
            except Exception as e:
                result[name] = {
                    'price': 0.0,
                    'change': 0.0,
                    'change_pct': 0.0,
                    'error': str(e),
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
    except Exception as general_e:
        st.error(f"Error fetching market indices: {str(general_e)}")
    
    return result

def fetch_latest_news(tickers=None, limit=10):
    """Fetch latest stock market news"""
    try:
        # If tickers provided, fetch news for those tickers
        if tickers and isinstance(tickers, list) and len(tickers) > 0:
            ticker_str = ",".join(tickers)
            url = f"https://api.polygon.io/v2/reference/news?ticker={ticker_str}&limit={limit}&sort=published_utc&order=desc&apiKey={POLYGON_API_KEY}"
        else:
            # Otherwise fetch general market news
            url = f"https://api.polygon.io/v2/reference/news?limit={limit}&sort=published_utc&order=desc&apiKey={POLYGON_API_KEY}"
        
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200 and 'results' in data:
            # Process news with sentiment analysis
            news_list = []
            for news in data['results']:
                # Get sentiment
                outcome_text, outcome_class = predict_news_outcome(news.get('title', ''))
                
                # Format date
                published_date = datetime.fromisoformat(news.get('published_utc', '').replace('Z', '+00:00'))
                formatted_date = published_date.strftime("%m-%d %H:%M")
                
                news_list.append({
                    'tickers': news.get('tickers', []),
                    'title': news.get('title', 'No headline available'),
                    'url': news.get('article_url', '#'),
                    'source': news.get('publisher', {}).get('name', 'Unknown'),
                    'date': formatted_date,
                    'sentiment': outcome_text,
                    'sentiment_class': outcome_class
                })
            
            return news_list
        else:
            st.error(f"Error fetching news: {data.get('error', 'Unknown error')}")
            return []
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

def predict_news_outcome(title):
    """Simple sentiment analysis to predict outcome based on headline"""
    positive_words = ['rise', 'jump', 'gain', 'surge', 'up', 'high', 'growth', 'profit', 
                     'beat', 'exceed', 'positive', 'bullish', 'rally', 'soar']
    negative_words = ['fall', 'drop', 'decline', 'down', 'low', 'loss', 'miss', 'below', 
                     'negative', 'bearish', 'plunge', 'sink', 'crash', 'struggle']
    
    title_lower = title.lower()
    
    positive_count = sum(1 for word in positive_words if re.search(r'\b' + word + r'\b', title_lower))
    negative_count = sum(1 for word in negative_words if re.search(r'\b' + word + r'\b', title_lower))
    
    if positive_count > negative_count:
        return "📈 Positive", "news-positive"
    elif negative_count > positive_count:
        return "📉 Negative", "news-negative"
    else:
        return "⟷ Neutral", "news-neutral"

def calculate_signals(ticker):
    """Calculate technical signals for a stock"""
    try:
        # Fetch historical data
        data = yf.download(
            tickers=ticker,
            period="60d",  # Enough for most indicators
            interval="1d",
            auto_adjust=True
        )
        
        if data.empty:
            return {"error": "No data available"}
        
        # Calculate moving averages
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Get latest values
        latest = data.iloc[-1]
        
        # Generate signals
        signals = {}
        
        # Moving Average Signal
        if latest['Close'] > latest['SMA20'] > latest['SMA50']:
            signals['MA'] = {"signal": "BUY", "message": "Price above both SMAs"}
        elif latest['SMA20'] < latest['SMA50'] and latest['Close'] < latest['SMA20']:
            signals['MA'] = {"signal": "SELL", "message": "Price below both SMAs"}
        else:
            signals['MA'] = {"signal": "HOLD", "message": "Mixed MA signals"}
        
        # RSI Signal
        if latest['RSI'] > 70:
            signals['RSI'] = {"signal": "SELL", "message": f"Overbought (RSI: {latest['RSI']:.1f})"}
        elif latest['RSI'] < 30:
            signals['RSI'] = {"signal": "BUY", "message": f"Oversold (RSI: {latest['RSI']:.1f})"}
        else:
            signals['RSI'] = {"signal": "HOLD", "message": f"Neutral (RSI: {latest['RSI']:.1f})"}
        
        # MACD Signal
        if latest['MACD'] > latest['Signal']:
            signals['MACD'] = {"signal": "BUY", "message": "MACD above signal line"}
        else:
            signals['MACD'] = {"signal": "SELL", "message": "MACD below signal line"}
        
        # Overall signal (simple majority)
        buy_count = sum(1 for sig in signals.values() if sig['signal'] == "BUY")
        sell_count = sum(1 for sig in signals.values() if sig['signal'] == "SELL")
        
        if buy_count > sell_count:
            overall = "BUY"
        elif sell_count > buy_count:
            overall = "SELL"
        else:
            overall = "HOLD"
        
        # Add overall signal and current values
        signals['OVERALL'] = {"signal": overall, "message": f"{buy_count} buy vs {sell_count} sell signals"}
        signals['values'] = {
            'price': latest['Close'],
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'signal': latest['Signal'],
            'sma20': latest['SMA20'],
            'sma50': latest['SMA50']
        }
        
        return signals
    except Exception as e:
        return {"error": str(e)}

# Function to refresh all data
def refresh_all_data():
    """Refresh all dashboard data"""
    st.session_state.refresh_counter += 1
    st.session_state.last_refresh = datetime.now()
    
    # Fetch stock data for watchlist
    st.session_state.stock_data = fetch_stock_data(st.session_state.watchlist)
    
    # Fetch market indices
    st.session_state.market_indices = fetch_market_indices()
    
    # Fetch news data - prioritize watchlist stocks
    st.session_state.news_data = fetch_latest_news(st.session_state.watchlist, 20)
    
    # Calculate signals for watchlist stocks
    signals = {}
    for ticker in st.session_state.watchlist:
        signals[ticker] = calculate_signals(ticker)
    st.session_state.signals_data = signals

# Functions to display different dashboard components
def display_watchlist_widget():
    """Display watchlist management widget"""
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header">Watchlist Manager</div>', unsafe_allow_html=True)
    
    # Add a stock to watchlist
    new_ticker = st.text_input("Add ticker:", key="add_ticker").upper()
    if st.button("Add", key="add_button") and new_ticker:
        if new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
            st.experimental_rerun()
    
    # Show current watchlist with remove buttons
    st.markdown('<div class="scrollable" style="max-height: 200px;">', unsafe_allow_html=True)
    for i, ticker in enumerate(st.session_state.watchlist):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"<div class='watchlist-item'>{ticker}</div>", unsafe_allow_html=True)
        with col2:
            if st.button("🗑️", key=f"remove_{i}"):
                st.session_state.watchlist.remove(ticker)
                st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Reset to default button
    if st.button("Reset to Default"):
        st.session_state.watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_stock_price_widget():
    """Display stock prices widget"""
    stock_data = st.session_state.stock_data
    
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header">Watchlist Prices</div>', unsafe_allow_html=True)
    
    if not stock_data:
        st.markdown("No stock data available.", unsafe_allow_html=True)
    else:
        for ticker, data in stock_data.items():
            if 'error' in data:
                st.markdown(f"<div>{ticker}: Error fetching data</div>", unsafe_allow_html=True)
                continue
                
            # Determine color based on price change
            color_class = "stock-up" if data['change'] >= 0 else "stock-down"
            change_sign = "+" if data['change'] >= 0 else ""
            
            # Create stock price display
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; padding-bottom: 5px; border-bottom: 1px solid #4a90e2;">
                <div style="font-weight: bold;">{ticker}</div>
                <div>${data['price']:.2f}</div>
                <div class="{color_class}">{change_sign}{data['change']:.2f} ({change_sign}{data['change_pct']:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_market_overview_widget():
    """Display market overview widget"""
    market_data = st.session_state.market_indices
    
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header">Market Overview</div>', unsafe_allow_html=True)
    
    if not market_data:
        st.markdown("No market data available.", unsafe_allow_html=True)
    else:
        for index, data in market_data.items():
            if 'error' in data:
                st.markdown(f"<div>{index}: Error fetching data</div>", unsafe_allow_html=True)
                continue
                
            # Determine color based on price change
            color_class = "stock-up" if data['change'] >= 0 else "stock-down"
            change_sign = "+" if data['change'] >= 0 else ""
            
            # Create market index display
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; padding-bottom: 5px; border-bottom: 1px solid #4a90e2;">
                <div style="font-weight: bold;">{index}</div>
                <div>{data['price']:.2f}</div>
                <div class="{color_class}">{change_sign}{data['change']:.2f} ({change_sign}{data['change_pct']:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_stock_charts_widget():
    """Display stock charts widget"""
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header">Stock Charts</div>', unsafe_allow_html=True)
    
    # Select stock for detailed chart
    selected_ticker = st.selectbox("Select stock:", st.session_state.watchlist)
    
    # Select timeframe
    timeframe = st.radio(
        "Timeframe:",
        ["1D", "5D", "1M"],
        horizontal=True
    )
    
    # Determine parameters based on timeframe
    if timeframe == "1D":
        interval = "5m"
        days = 1
    elif timeframe == "5D":
        interval = "15m"
        days = 5
    else:  # 1M
        interval = "1d"
        days = 30
    
    # Fetch and display chart
    data = fetch_stock_intraday(selected_ticker, interval, days)
    
    if not data.empty:
        # Create price chart
        fig = px.line(
            data, 
            x='Datetime', 
            y='Close',
            title=f"{selected_ticker} - {timeframe}"
        )
        
        # Add volume as bar chart on secondary y-axis
        fig.add_trace(
            go.Bar(
                x=data['Datetime'],
                y=data['Volume'],
                name='Volume',
                marker=dict(color='rgba(74, 144, 226, 0.3)'),
                opacity=0.3,
                yaxis="y2"
            )
        )
        
        # Update layout for terminal theme
        fig.update_layout(
            paper_bgcolor='#2f2f2f',
            plot_bgcolor='#2f2f2f',
            font=dict(color='#ffffff'),
            title_font=dict(color='#e6f3ff'),
            xaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff'),
            yaxis=dict(
                gridcolor='#4a90e2', 
                linecolor='#e6f3ff', 
                zerolinecolor='#e6f3ff', 
                title='Price'
            ),
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400
        )
        
        fig.update_traces(
            line_color='#e6f3ff',
            selector=dict(type='scatter')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display price stats
        if len(data) > 0:
            current = data['Close'].iloc[-1]
            high = data['High'].max()
            low = data['Low'].min()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current", f"${current:.2f}")
            with col2:
                st.metric("High", f"${high:.2f}")
            with col3:
                st.metric("Low", f"${low:.2f}")
    else:
        st.warning(f"No data available for {selected_ticker}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_news_widget():
    """Display news widget"""
    news_data = st.session_state.news_data
    
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header">Latest Market News</div>', unsafe_allow_html=True)
    
    # Filter options
    news_filter = st.selectbox(
        "Filter by:",
        ["All News", "Positive Only", "Negative Only", "Neutral Only"],
        key="news_filter"
    )
    
    if not news_data:
        st.markdown("No news available.", unsafe_allow_html=True)
    else:
        st.markdown('<div class="scrollable">', unsafe_allow_html=True)
        
        for news in news_data:
            # Filter based on user selection
            if (news_filter == "Positive Only" and news['sentiment_class'] != "news-positive") or \
               (news_filter == "Negative Only" and news['sentiment_class'] != "news-negative") or \
               (news_filter == "Neutral Only" and news['sentiment_class'] != "news-neutral"):
                continue
            
            # Get tickers for this news item
            ticker_str = ", ".join(news['tickers']) if news['tickers'] else "General"
            
            # Display news item
            st.markdown(f"""
            <div class="news-item">
                <div style="display: flex; justify-content: space-between;">
                    <span class="news-ticker">{ticker_str}</span>
                    <span>{news['date']}</span>
                </div>
                <a href="{news['url']}" target="_blank" class="news-headline">{news['title']}</a>
                <div style="display: flex; justify-content: space-between;">
                    <span>{news['source']}</span>
                    <span class="news-outcome {news['sentiment_class']}">{news['sentiment']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_signals_widget():
    """Display technical signals widget"""
    signals_data = st.session_state.signals_data
    
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header">Technical Signals</div>', unsafe_allow_html=True)
    
    # Select stock for signals
    selected_ticker = st.selectbox("Select stock:", st.session_state.watchlist, key="signals_ticker")
    
    if selected_ticker not in signals_data:
        st.markdown("No signal data available.", unsafe_allow_html=True)
    elif 'error' in signals_data[selected_ticker]:
        st.markdown(f"Error: {signals_data[selected_ticker]['error']}", unsafe_allow_html=True)
    else:
        signals = signals_data[selected_ticker]
        
        # Display overall signal first
        overall = signals['OVERALL']
        signal_class = "signal-buy" if overall['signal'] == "BUY" else "signal-sell" if overall['signal'] == "SELL" else "signal-hold"
        
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 15px;">
            <div style="font-size: 24px; font-weight: bold;">Overall Signal</div>
            <div class="signal-badge {signal_class}" style="font-size: 26px; padding: 10px 20px;">{overall['signal']}</div>
            <div>{overall['message']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display individual signals
        st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
        
        for key, signal in signals.items():
            if key in ['OVERALL', 'values']:
                continue
                
            signal_class = "signal-buy" if signal['signal'] == "BUY" else "signal-sell" if signal['signal'] == "SELL" else "signal-hold"
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; padding-bottom: 5px; border-bottom: 1px solid #4a90e2;">
                <div style="font-weight: bold;">{key}</div>
                <div class="signal-badge {signal_class}">{signal['signal']}</div>
                <div>{signal['message']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display current values
        if 'values' in signals:
            values = signals['values']
            st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #e6f3ff;'>Current Values</h4>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Price", f"${values['price']:.2f}")
                st.metric("SMA20", f"${values['sma20']:.2f}")
            with col2:
                st.metric("RSI", f"{values['rsi']:.1f}")
                st.metric("SMA50", f"${values['sma50']:.2f}")
            with col3:
                st.metric("MACD", f"{values['macd']:.3f}")
                st.metric("Signal Line", f"{values['signal']:.3f}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_dashboard_settings():
    """Display dashboard settings sidebar"""
    st.sidebar.markdown("<h3 style='color: #e6f3ff;'>Dashboard Settings</h3>", unsafe_allow_html=True)
    
    # Enable/disable auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
        refresh_all_data()
        st.experimental_rerun()
    
    # Customize layout
    st.sidebar.markdown("<h4 style='color: #e6f3ff; margin-top: 20px;'>Customize Layout</h4>", unsafe_allow_html=True)
    
    layout_changed = False
    
    for widget, current_value in st.session_state.layout.items():
        display_name = " ".join(word.capitalize() for word in widget.split("_"))
        new_value = st.sidebar.checkbox(f"Show {display_name}", value=current_value)
        
        if new_value != current_value:
            st.session_state.layout[widget] = new_value
            layout_changed = True
    
    # If layout changed, rerun to update
    if layout_changed:
        st.experimental_rerun()
    
    return auto_refresh

def main():
    """Main function to run the application"""
    # Apply terminal CSS
    st.markdown(terminal_css, unsafe_allow_html=True)
    
    # Title with terminal styling
    st.markdown("<h1 style='text-align: center; color: #e6f3ff;'>🐱 Meow Dashboard 🐱</h1>", unsafe_allow_html=True)
    
    # Display dashboard settings in sidebar
    auto_refresh = display_dashboard_settings()
    
    # Initial data fetch if empty
    if not st.session_state.stock_data:
        refresh_all_data()
    
    # Auto-refresh logic
    if auto_refresh:
        # Check if 5 seconds have passed since last refresh
        time_diff = (datetime.now() - st.session_state.last_refresh).total_seconds()
        if time_diff >= 5:
            refresh_all_data()
            st.experimental_rerun()
        
        # Display refresh indicator
        st.markdown(f"""
        <div class="refresh-indicator">
            Auto-refresh: {5 - int(time_diff)}s | Last: {st.session_state.last_refresh.strftime('%H:%M:%S')} | Count: {st.session_state.refresh_counter}
        </div>
        """, unsafe_allow_html=True)
    
    # Main dashboard layout - using custom column system for more flexibility
    st.markdown('<div class="row">', unsafe_allow_html=True)
    
    # Left column (25%)
    st.markdown('<div class="custom-column col-25">', unsafe_allow_html=True)
    
    # Watchlist manager
    if st.session_state.layout['watchlist']:
        display_watchlist_widget()
    
    # Stock prices
    if st.session_state.layout['watchlist']:
        display_stock_price_widget()
    
    # Market overview
    if st.session_state.layout['market_overview']:
        display_market_overview_widget()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Middle column (50%)
    st.markdown('<div class="custom-column col-50">', unsafe_allow_html=True)
    
    # Stock charts
    if st.session_state.layout['stock_charts']:
        display_stock_charts_widget()
    
    # News feed
    if st.session_state.layout['news']:
        display_news_widget()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Right column (25%)
    st.markdown('<div class="custom-column col-25">', unsafe_allow_html=True)
    
    # Technical signals
    if st.session_state.layout['signals']:
        display_signals_widget()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # End row
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="margin-top: 30px; text-align: center; color: #e6f3ff; font-size: 12px;">
        <p>© 2025 Meow Terminal | Dashboard Module</p>
    </div>
    """, unsafe_allow_html=True)

# This is the entry point of the script
if __name__ == "__main__":
    main()
