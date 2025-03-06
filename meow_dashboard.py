import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import re
import time
from datetime import datetime, timedelta
import yfinance as yf

# Polygon API key
POLYGON_API_KEY = "9skphQ6G7_rESW6iTNJDIAycT9gncpje"

# Set page config
st.set_page_config(
    page_title="Meow Dashboard",
    page_icon="😺",
    layout="wide",
)

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
    
    div.stMetric label {
        color: #e6f3ff !important;
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
    
    /* Dashboard widget */
    .dashboard-widget {
        border: 1px solid #4a90e2;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
        background-color: #2f2f2f;
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
    
    /* Stock price info */
    .stock-up {
        color: #4CAF50 !important;
    }
    
    .stock-down {
        color: #F44336 !important;
    }
    
    /* Scrollable container */
    .scrollable {
        max-height: 300px;
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
    
    /* Signal badge */
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
    
    /* News item styling */
    .news-item {
        background-color: #2f2f2f;
        border: 1px solid #4a90e2;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    
    .news-ticker {
        font-weight: bold;
        color: #e6f3ff;
        font-size: 16px;
    }
    
    .news-headline {
        font-size: 18px;
        margin: 5px 0;
        color: #ffffff;
        text-decoration: none;
    }
    
    .news-headline:hover {
        text-decoration: underline;
        cursor: pointer;
    }
    
    .news-outcome {
        font-weight: bold;
        font-size: 16px;
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
    
    /* Data tables */
    .dataframe {
        width: 100%;
    }
    
    .dataframe th {
        background-color: #4a90e2;
        color: white;
        padding: 8px;
        text-align: left;
    }
    
    .dataframe td {
        padding: 8px;
        border-bottom: 1px solid rgba(74, 144, 226, 0.3);
    }
    
    /* Calendar event styling */
    .calendar-event {
        padding: 5px;
        margin-bottom: 5px;
        border-left: 3px solid #4a90e2;
        background-color: rgba(74, 144, 226, 0.1);
    }
    
    .calendar-event-earnings {
        border-left-color: #4CAF50;
    }
    
    .calendar-event-dividend {
        border-left-color: #FFC107;
    }
    
    .calendar-event-economic {
        border-left-color: #F44336;
    }
    
    /* Portfolio styling */
    .portfolio-summary {
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid #4a90e2;
        font-weight: bold;
    }
</style>
"""

# Function to create a Windows 95-style header
def win95_header(text):
    return f'<div class="win95-header">{text}</div>'

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
                    if len(tickers_data) >= 2:
                        current_price = tickers_data['Close'].iloc[-1]
                        prev_price = tickers_data['Close'].iloc[-2]
                    else:
                        current_price = tickers_data['Close'].iloc[-1]
                        prev_price = tickers_data['Open'].iloc[-1]
                else:
                    # Multiple tickers case
                    if len(tickers_data[ticker]['Close']) >= 2:
                        current_price = tickers_data[ticker]['Close'].iloc[-1]
                        prev_price = tickers_data[ticker]['Close'].iloc[-2]
                    else:
                        current_price = tickers_data[ticker]['Close'].iloc[-1]
                        prev_price = tickers_data[ticker]['Open'].iloc[-1]
                
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
                    if len(data[ticker]['Close']) >= 2:
                        current_price = data[ticker]['Close'].iloc[-1]
                        prev_price = data[ticker]['Close'].iloc[-2]
                    else:
                        current_price = data[ticker]['Close'].iloc[-1]
                        prev_price = data[ticker]['Open'].iloc[-1]
                else:  # Single ticker
                    if len(data['Close']) >= 2:
                        current_price = data['Close'].iloc[-1]
                        prev_price = data['Close'].iloc[-2]
                    else:
                        current_price = data['Close'].iloc[-1]
                        prev_price = data['Open'].iloc[-1]
                
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

def fetch_latest_news(tickers=None, limit=15):
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

def fetch_economic_calendar():
    """Generate a simulated economic calendar"""
    today = datetime.now()
    
    # Generate next 7 days
    calendar = []
    
    # Sample events
    events = [
        {"type": "earnings", "title": "AAPL Earnings", "date": today + timedelta(days=2)},
        {"type": "earnings", "title": "MSFT Earnings", "date": today + timedelta(days=4)},
        {"type": "economic", "title": "Fed Rate Decision", "date": today + timedelta(days=3)},
        {"type": "economic", "title": "Jobs Report", "date": today + timedelta(days=5)},
        {"type": "dividend", "title": "JNJ Ex-Dividend", "date": today + timedelta(days=1)},
        {"type": "economic", "title": "CPI Data Release", "date": today + timedelta(days=6)},
        {"type": "earnings", "title": "GOOGL Earnings", "date": today + timedelta(days=3)},
        {"type": "dividend", "title": "PG Ex-Dividend", "date": today + timedelta(days=2)}
    ]
    
    # Sort by date
    events.sort(key=lambda x: x["date"])
    
    return events[:7]  # Return the closest 7 events

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

def refresh_all_data():
    """Refresh all dashboard data"""
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
    
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
    
    # Fetch economic calendar
    st.session_state.calendar_data = fetch_economic_calendar()
    
    # Update portfolio values
    if 'portfolio' in st.session_state:
        for ticker, position in st.session_state.portfolio.items():
            if ticker in st.session_state.stock_data:
                current_price = st.session_state.stock_data[ticker]['price']
                position['current_price'] = current_price
                position['current_value'] = current_price * position['shares']
                position['gain_loss'] = position['current_value'] - position['cost_basis']
                position['gain_loss_pct'] = (position['gain_loss'] / position['cost_basis']) * 100 if position['cost_basis'] > 0 else 0

def initialize_demo_portfolio():
    """Initialize a demo portfolio for visualization"""
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            "AAPL": {
                "shares": 10,
                "avg_price": 175.50,
                "cost_basis": 1755.00,
                "current_price": 0.0,
                "current_value": 0.0,
                "gain_loss": 0.0,
                "gain_loss_pct": 0.0
            },
            "MSFT": {
                "shares": 5,
                "avg_price": 350.25,
                "cost_basis": 1751.25,
                "current_price": 0.0,
                "current_value": 0.0,
                "gain_loss": 0.0,
                "gain_loss_pct": 0.0
            },
            "TSLA": {
                "shares": 8,
                "avg_price": 220.75,
                "cost_basis": 1766.00,
                "current_price": 0.0,
                "current_value": 0.0,
                "gain_loss": 0.0,
                "gain_loss_pct": 0.0
            }
        }

def get_portfolio_summary():
    """Calculate portfolio summary statistics"""
    if 'portfolio' not in st.session_state:
        return {
            'total_value': 0.0,
            'total_cost': 0.0,
            'total_gain_loss': 0.0,
            'total_gain_loss_pct': 0.0
        }
    
    portfolio = st.session_state.portfolio
    
    # Calculate totals
    total_value = sum(pos['current_value'] for pos in portfolio.values())
    total_cost = sum(pos['cost_basis'] for pos in portfolio.values())
    total_gain_loss = total_value - total_cost
    total_gain_loss_pct = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0
    
    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_gain_loss': total_gain_loss,
        'total_gain_loss_pct': total_gain_loss_pct
    }

def display_watchlist_manager():
    """Display watchlist management widget"""
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header">Watchlist Manager</div>', unsafe_allow_html=True)
    
    # Add a stock to watchlist
    new_ticker = st.text_input("Add ticker:", key="add_ticker").upper()
    if st.button("Add", key="add_button") and new_ticker:
        if new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
            st.rerun()
    
    # Show current watchlist with remove buttons
    st.markdown('<div class="scrollable">', unsafe_allow_html=True)
    for i, ticker in enumerate(st.session_state.watchlist):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"<div style='padding: 5px; border-bottom: 1px solid #4a90e2;'>{ticker}</div>", unsafe_allow_html=True)
        with col2:
            if st.button("🗑️", key=f"remove_{i}"):
                st.session_state.watchlist.remove(ticker)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Reset to default button
    if st.button("Reset to Default"):
        st.session_state.watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_stock_price_widget():
    """Display stock prices widget"""
    stock_data = st.session_state.stock_data
    
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header">Watchlist Prices</div>', unsafe_allow_html=True)
    
    if not stock_data:
        st.markdown("No stock data available.", unsafe_allow_html=True)
    else:
        stock_table_html = """
        <table class="dataframe" style="width:100%">
            <tr>
                <th>Ticker</th>
                <th>Price</th>
                <th>Change</th>
                <th>Change %</th>
            </tr>
        """
        
        for ticker, data in stock_data.items():
            if 'error' in data:
                stock_table_html += f"""
                <tr>
                    <td>{ticker}</td>
                    <td colspan="3">Error fetching data</td>
                </tr>
                """
                continue
                
            # Determine color based on price change
            color_class = "stock-up" if data['change'] >= 0 else "stock-down"
            change_sign = "+" if data['change'] >= 0 else ""
            
            stock_table_html += f"""
            <tr>
                <td><b>{ticker}</b></td>
                <td>${data['price']:.2f}</td>
                <td class="{color_class}">{change_sign}{data['change']:.2f}</td>
                <td class="{color_class}">{change_sign}{data['change_pct']:.2f}%</td>
            </tr>
            """
        
        stock_table_html += "</table>"
        st.markdown(stock_table_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_market_overview_widget():
    """Display market overview widget"""
    market_data = st.session_state.market_indices
    
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header">Market Overview</div>', unsafe_allow_html=True)
    
    if not market_data:
        st.markdown("No market data available.", unsafe_allow_html=True)
    else:
        # Create columns for each index
        cols = st.columns(len(market_data))
        
        for i, (index, data) in enumerate(market_data.items()):
            with cols[i]:
                if 'error' in data:
                    st.error(f"{index}: Error fetching data")
                else:
                    # Determine color based on price change
                    delta_color = "normal" if data['change_pct'] >= 0 else "inverse"
                    delta = f"{'+' if data['change_pct'] >= 0 else ''}{data['change_pct']:.2f}%"
                    
                    st.metric(
                        label=index,
                        value=f"{data['price']:.2f}",
                        delta=delta,
                        delta_color=delta_color
                    )
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_portfolio_widget():
    """Display portfolio tracking widget"""
    # Initialize demo portfolio
    initialize_demo_portfolio()
    portfolio = st.session_state.portfolio
    
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header">Portfolio Tracker</div>', unsafe_allow_html=True)
    
    # Get portfolio summary
    summary = get_portfolio_summary()
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Value", f"${summary['total_value']:.2f}")
    with col2:
        st.metric("Total Cost", f"${summary['total_cost']:.2f}")
    with col3:
        delta = f"{'+' if summary['total_gain_loss'] >= 0 else ''}{summary['total_gain_loss_pct']:.2f}%"
        delta_color = "normal" if summary['total_gain_loss'] >= 0 else "inverse"
        st.metric("Gain/Loss", f"${summary['total_gain_loss']:.2f}", delta=delta, delta_color=delta_color)
    
    # Display portfolio table
    portfolio_table_html = """
    <table class="dataframe" style="width:100%">
        <tr>
            <th>Ticker</th>
            <th>Shares</th>
            <th>Avg Price</th>
            <th>Current</th>
            <th>Value</th>
            <th>Gain/Loss</th>
        </tr>
    """
    
    for ticker, position in portfolio.items():
        # Determine color based on gain/loss
        color_class = "stock-up" if position['gain_loss'] >= 0 else "stock-down"
        change_sign = "+" if position['gain_loss'] >= 0 else ""
        
        portfolio_table_html += f"""
        <tr>
            <td><b>{ticker}</b></td>
            <td>{position['shares']}</td>
            <td>${position['avg_price']:.2f}</td>
            <td>${position['current_price']:.2f}</td>
            <td>${position['current_value']:.2f}</td>
            <td class="{color_class}">{change_sign}${position['gain_loss']:.2f} ({change_sign}{position['gain_loss_pct']:.2f}%)</td>
        </tr>
        """
    
    portfolio_table_html += "</table>"
    st.markdown(portfolio_table_html, unsafe_allow_html=True)
    
    # Add position option
    with st.expander("Add New Position"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_ticker = st.text_input("Ticker", key="portfolio_ticker").upper()
        with col2:
            new_shares = st.number_input("Shares", min_value=0.01, step=1.0, key="portfolio_shares")
        with col3:
            new_price = st.number_input("Avg Price ($)", min_value=0.01, step=1.0, key="portfolio_price")
        
        if st.button("Add Position", key="add_position_button"):
            if new_ticker and new_shares > 0 and new_price > 0:
                cost_basis = new_shares * new_price
                
                # Add to portfolio
                st.session_state.portfolio[new_ticker] = {
                    "shares": new_shares,
                    "avg_price": new_price,
                    "cost_basis": cost_basis,
                    "current_price": 0.0,
                    "current_value": 0.0,
                    "gain_loss": 0.0,
                    "gain_loss_pct": 0.0
                }
                
                # Add to watchlist if not already there
                if new_ticker not in st.session_state.watchlist:
                    st.session_state.watchlist.append(new_ticker)
                
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_stock_charts_widget():
    """Display stock charts widget"""
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header">Stock Charts</div>', unsafe_allow_html=True)
    
    # Select stock for detailed chart
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_ticker = st.selectbox("Select stock:", st.session_state.watchlist, key="chart_ticker")
    
    with col2:
        # Select timeframe
        timeframe = st.radio(
            "Timeframe:",
            ["1D", "5D", "1M"],
            horizontal=True,
            key="chart_timeframe"
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
        st.markdown("No news available. Please refresh the dashboard.", unsafe_allow_html=True)
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
        st.markdown("No signal data available. Please refresh the dashboard.", unsafe_allow_html=True)
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
        
        signals_table_html = """
        <table class="dataframe" style="width:100%">
            <tr>
                <th>Indicator</th>
                <th>Signal</th>
                <th>Message</th>
            </tr>
        """
        
        for key, signal in signals.items():
            if key in ['OVERALL', 'values']:
                continue
                
            signal_class = "signal-buy" if signal['signal'] == "BUY" else "signal-sell" if signal['signal'] == "SELL" else "signal-hold"
            
            signals_table_html += f"""
            <tr>
                <td><b>{key}</b></td>
                <td><span class="signal-badge {signal_class}">{signal['signal']}</span></td>
                <td>{signal['message']}</td>
            </tr>
            """
        
        signals_table_html += "</table>"
        st.markdown(signals_table_html, unsafe_allow_html=True)
        
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

def display_calendar_widget():
    """Display economic calendar widget"""
    calendar_data = st.session_state.calendar_data
    
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header">Economic Calendar</div>', unsafe_allow_html=True)
    
    if not calendar_data:
        st.markdown("No calendar data available.", unsafe_allow_html=True)
    else:
        for event in calendar_data:
            event_type_class = f"calendar-event-{event['type']}"
            event_date = event['date'].strftime("%a, %b %d")
            
            st.markdown(f"""
            <div class="calendar-event {event_type_class}">
                <div style="display: flex; justify-content: space-between;">
                    <span><b>{event['title']}</b></span>
                    <span>{event_date}</span>
                </div>
                <div>Type: {event['type'].capitalize()}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main function to run the dashboard"""
    # Apply terminal CSS
    st.markdown(terminal_css, unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if 'layout' not in st.session_state:
        st.session_state.layout = {
            'watchlist': True,
            'market_overview': True,
            'stock_charts': True,
            'news': True,
            'signals': True,
            'portfolio': True,
            'calendar': True
        }
    
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = {}
    
    if 'market_indices' not in st.session_state:
        st.session_state.market_indices = {}
    
    if 'news_data' not in st.session_state:
        st.session_state.news_data = []
    
    if 'signals_data' not in st.session_state:
        st.session_state.signals_data = {}
    
    if 'calendar_data' not in st.session_state:
        st.session_state.calendar_data = []
    
    # Title with terminal styling
    st.markdown("<h1 style='text-align: center; color: #e6f3ff;'>🐱 Meow Dashboard 🐱</h1>", unsafe_allow_html=True)
    
    # Split dashboard into sidebar and main content
    col_settings, col_content = st.columns([1, 3])
    
    with col_settings:
        # Dashboard settings
        st.markdown("<h3 style='color: #e6f3ff;'>Dashboard Settings</h3>", unsafe_allow_html=True)
        
        # Enable/disable auto-refresh
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
        
        # Manual refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            refresh_all_data()
            st.rerun()
        
        # Customize layout
        st.markdown("<h3 style='color: #e6f3ff; margin-top: 20px;'>Layout Settings</h3>", unsafe_allow_html=True)
        
        layout_changed = False
        
        for widget, current_value in st.session_state.layout.items():
            display_name = " ".join(word.capitalize() for word in widget.split("_"))
            new_value = st.checkbox(f"Show {display_name}", value=current_value, key=f"layout_{widget}")
            
            if new_value != current_value:
                st.session_state.layout[widget] = new_value
                layout_changed = True
        
        # Watchlist management
        if st.session_state.layout['watchlist']:
            display_watchlist_manager()
    
    with col_content:
        # Initial data fetch if empty
        if not st.session_state.stock_data:
            refresh_all_data()
        
        # Auto-refresh logic
        if auto_refresh:
            # Check if 5 seconds have passed since last refresh
            time_diff = (datetime.now() - st.session_state.last_refresh).total_seconds()
            if time_diff >= 5:
                refresh_all_data()
                st.rerun()
            
            # Display refresh indicator
            st.markdown(f"""
            <div class="refresh-indicator">
                Auto-refresh: {5 - int(time_diff)}s | Last: {st.session_state.last_refresh.strftime('%H:%M:%S')} | Count: {st.session_state.refresh_counter}
            </div>
            """, unsafe_allow_html=True)
        
        # Market overview at the top
        if st.session_state.layout['market_overview']:
            display_market_overview_widget()
        
        # Main dashboard layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Stock prices
            if st.session_state.layout['watchlist']:
                display_stock_price_widget()
            
            # News feed
            if st.session_state.layout['news']:
                display_news_widget()
            
            # Calendar
            if st.session_state.layout['calendar']:
                display_calendar_widget()
            
        with col2:
            # Portfolio
            if st.session_state.layout['portfolio']:
                display_portfolio_widget()
            
            # Stock charts
            if st.session_state.layout['stock_charts']:
                display_stock_charts_widget()
            
            # Technical signals
            if st.session_state.layout['signals']:
                display_signals_widget()
        
    # Footer
    st.markdown("""
    <div style="margin-top: 20px; text-align: center; color: #e6f3ff; font-size: 12px;">
        <p>© 2025 Meow Terminal | Dashboard Module</p>
    </div>
    """, unsafe_allow_html=True)

# This is the entry point of the script
if __name__ == "__main__":
    main()
