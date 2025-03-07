def display_stock_price_widget():
    """Display enhanced stock prices widget with more data"""
    stock_data = st.session_state.stock_data
    
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header"><span class="live-indicator"></span>Watchlist Prices <span class="new-feature">Live</span></div>', unsafe_allow_html=True)
    
    if not stock_data:
        st.markdown("No stock data available.", unsafe_allow_html=True)
    else:
        # Filter to show only watchlist stocks based on active watchlist
        active_watchlist = st.session_state.get('active_watchlist', 'primary')
        current_list = st.session_state.watchlist if active_watchlist == 'primary' else st.session_state.secondary_watchlist
        
        # Add search filter
        search = st.text_input("Search tickers:", key="stock_search")
        
        # Add sort options
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox(
                "Sort by:",
                ["Ticker", "Price", "% Change"],
                key="price_sort_by"
            )
        with col2:
            sort_order = st.radio(
                "Order:",
                ["Ascending", "Descending"],
                horizontal=True,
                key="price_sort_order"
            )
        
        # Filter and sort data
        filtered_data = []
        for ticker, data in stock_data.items():
            if ticker in current_list and 'error' not in data:
                if search and search.upper() not in ticker:
                    continue
                filtered_data.append((ticker, data))
        
        # Sort data
        if sort_by == "Ticker":
            filtered_data.sort(key=lambda x: x[0])
        elif sort_by == "Price":
            filtered_data.sort(key=lambda x: x[1]['price'])
        elif sort_by == "% Change":
            filtered_data.sort(key=lambda x: x[1]['change_pct'])
        
        # Apply sort order
        if sort_order == "Descending":
            filtered_data.reverse()
        
        # Display in an enhanced table with more data
        st.markdown("""
        <div class="scrollable" style="max-height: 400px;">
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="border-bottom: 2px solid #4a90e2;">
                        <th style="text-align: left; padding: 8px;">Ticker</th>
                        <th style="text-align: right; padding: 8px;">Price</th>
                        <th style="text-align: right; padding: 8px;">Change</th>
                        <th style="text-align: right; padding: 8px;">% Change</th>
                        <th style="text-align: right; padding: 8px;">Updated</th>
                    </tr>
                </thead>
                <tbody>
        """, unsafe_allow_html=True)
        
        for ticker, data in filtered_data:
            # Determine color based on price change
            color_class = "stock-up" if data['change'] >= 0 else "stock-down"
            change_sign = "+" if data['change'] >= 0 else ""
            
            # Create stock price row
            st.markdown(f"""
            <tr style="border-bottom: 1px solid #4a90e2;">
                <td style="text-align: left; padding: 8px; font-weight: bold;">{ticker}</td>
                <td style="text-align: right; padding: 8px;">${data['price']:.2f}</td>
                <td style="text-align: right; padding: 8px;" class="{color_class}">{change_sign}{data['change']:.2f}</td>
                <td style="text-align: right; padding: 8px;" class="{color_class}">{change_sign}{data['change_pct']:.2f}%</td>
                <td style="text-align: right; padding: 8px; font-size: 12px;">{data['timestamp']}</td>
            </tr>
            """, unsafe_allow_html=True)
        
        if not filtered_data:
            st.markdown("""
            <tr>
                <td colspan="5" style="text-align: center; padding: 20px;">No matching stocks found</td>
            </tr>
            """, unsafe_allow_html=True)
        
        st.markdown("""
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)import streamlit as st
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
    page_title="Meow Dashboard",  # Already correct
    page_icon="😺",
    layout="wide",
)

# Polygon API key
POLYGON_API_KEY = "9skphQ6G7_rESW6iTNJDIAycT9gncpje"

# Apply terminal-style CSS with enhanced styling
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
    
    /* Popup/Modal styling */
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.7);
        z-index: 1000;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .modal-container {
        background-color: #2f2f2f;
        border: 2px solid #4a90e2;
        border-radius: 5px;
        width: 80%;
        max-width: 800px;
        max-height: 80vh;
        overflow-y: auto;
        padding: 20px;
        position: relative;
    }
    
    .modal-close {
        position: absolute;
        top: 10px;
        right: 10px;
        cursor: pointer;
        font-size: 20px;
        color: #e6f3ff;
    }
    
    /* Draggable mini-window */
    .mini-window {
        position: fixed;
        background-color: #2f2f2f;
        border: 2px solid #4a90e2;
        border-radius: 5px;
        min-width: 300px;
        min-height: 200px;
        z-index: 999;
        resize: both;
        overflow: auto;
    }
    
    .mini-window-header {
        background-color: #4a90e2;
        color: white;
        padding: 5px 10px;
        cursor: move;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .mini-window-content {
        padding: 10px;
        height: calc(100% - 30px);
        overflow: auto;
    }
    
    /* Alert levels */
    .alert-high {
        background-color: rgba(244, 67, 54, 0.3);
        border-left: 4px solid #F44336;
        padding: 8px;
        margin: 5px 0;
    }
    
    .alert-medium {
        background-color: rgba(255, 193, 7, 0.3);
        border-left: 4px solid #FFC107;
        padding: 8px;
        margin: 5px 0;
    }
    
    .alert-low {
        background-color: rgba(76, 175, 80, 0.3);
        border-left: 4px solid #4CAF50;
        padding: 8px;
        margin: 5px 0;
    }
    
    /* New feature badges */
    .new-feature {
        background-color: #4a90e2;
        color: white;
        padding: 2px 6px;
        border-radius: 10px;
        font-size: 10px;
        margin-left: 5px;
    }
    
    /* Live data indicator */
    .live-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #4CAF50;
        margin-right: 5px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            opacity: 1;
        }
        50% {
            opacity: 0.3;
        }
        100% {
            opacity: 1;
        }
    }
    
    /* Tab styling */
    .custom-tabs {
        display: flex;
        border-bottom: 1px solid #4a90e2;
        margin-bottom: 10px;
    }
    
    .custom-tab {
        padding: 8px 16px;
        cursor: pointer;
        background-color: #2f2f2f;
        border-top: 1px solid #4a90e2;
        border-left: 1px solid #4a90e2;
        border-right: 1px solid #4a90e2;
        border-bottom: none;
        margin-right: 2px;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
    }
    
    .custom-tab.active {
        background-color: #4a90e2;
        color: white;
    }
    
    /* Custom toggle switch */
    .toggle-container {
        display: flex;
        align-items: center;
    }
    
    .toggle-switch {
        position: relative;
        display: inline-block;
        width: 40px;
        height: 20px;
        margin: 0 8px;
    }
    
    .toggle-switch input {
        opacity: 0;
        width: 0;
        height: 0;
    }
    
    .toggle-slider {
        position: absolute;
        cursor: pointer;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: #2f2f2f;
        border: 1px solid #4a90e2;
        transition: .4s;
        border-radius: 20px;
    }
    
    .toggle-slider:before {
        position: absolute;
        content: "";
        height: 16px;
        width: 16px;
        left: 2px;
        bottom: 1px;
        background-color: #4a90e2;
        transition: .4s;
        border-radius: 50%;
    }
    
    input:checked + .toggle-slider {
        background-color: #2f2f2f;
    }
    
    input:checked + .toggle-slider:before {
        transform: translateX(19px);
        background-color: #4CAF50;
    }
    
    /* Top performing stocks table */
    .performance-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .performance-table th {
        background-color: #4a90e2;
        color: white;
        padding: 5px;
        text-align: left;
    }
    
    .performance-table td {
        padding: 5px;
        border-bottom: 1px solid #4a90e2;
    }
</style>

<script>
// JavaScript for mini-window functionality
document.addEventListener('DOMContentLoaded', function() {
    // Make mini-windows draggable
    const miniWindows = document.querySelectorAll('.mini-window');
    
    miniWindows.forEach(window => {
        const header = window.querySelector('.mini-window-header');
        
        let isDragging = false;
        let offsetX, offsetY;
        
        header.addEventListener('mousedown', function(e) {
            isDragging = true;
            offsetX = e.clientX - window.getBoundingClientRect().left;
            offsetY = e.clientY - window.getBoundingClientRect().top;
            
            window.style.opacity = '0.8';
        });
        
        document.addEventListener('mousemove', function(e) {
            if(isDragging) {
                window.style.left = (e.clientX - offsetX) + 'px';
                window.style.top = (e.clientY - offsetY) + 'px';
            }
        });
        
        document.addEventListener('mouseup', function() {
            isDragging = false;
            window.style.opacity = '1';
        });
        
        // Close button functionality
        const closeBtn = window.querySelector('.mini-window-close');
        if(closeBtn) {
            closeBtn.addEventListener('click', function() {
                window.style.display = 'none';
            });
        }
    });
});
</script>
"""

# Function to create a Windows 95-style header
def win95_header(text):
    return f'<div class="win95-header">{text}</div>'

# Session state initialization
if 'watchlist' not in st.session_state:
    # Default watchlist with more stocks
    st.session_state.watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD"]

if 'secondary_watchlist' not in st.session_state:
    # Secondary watchlist for separate tracking
    st.session_state.secondary_watchlist = ["JPM", "BAC", "C", "WFC", "GS"]

if 'layout' not in st.session_state:
    st.session_state.layout = {
        'watchlist': True,
        'market_overview': True,
        'stock_charts': True,
        'news': True,
        'signals': True,
        'performance': True,  # New widget
        'alerts': True,       # New widget
        'economic_data': True # New widget
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

if 'alerts' not in st.session_state:
    # Sample alerts
    st.session_state.alerts = [
        {"level": "high", "text": "AAPL breached 52-week high", "time": "10:30 AM"},
        {"level": "medium", "text": "MSFT trading volume 50% above average", "time": "11:45 AM"},
        {"level": "low", "text": "Market volatility index decreasing", "time": "01:15 PM"}
    ]

if 'refresh_rate' not in st.session_state:
    # Default refresh rate in seconds
    st.session_state.refresh_rate = 5

if 'open_windows' not in st.session_state:
    # Track open mini windows
    st.session_state.open_windows = {
        'watchlist': False,
        'alerts': False
    }

if 'custom_themes' not in st.session_state:
    # Predefined custom themes
    st.session_state.custom_themes = {
        'Terminal Blue': {
            'background': '#2f2f2f',
            'text': '#ffffff',
            'accent': '#4a90e2',
            'positive': '#4CAF50',
            'negative': '#F44336'
        },
        'Dark Forest': {
            'background': '#1e2a20',
            'text': '#e0e0e0',
            'accent': '#3e7b4f',
            'positive': '#6abf69',
            'negative': '#e57373'
        },
        'Midnight Purple': {
            'background': '#1a1a2e',
            'text': '#e0e0e0',
            'accent': '#7b2cbf',
            'positive': '#72b01d',
            'negative': '#ff5c8d'
        }
    }

if 'current_theme' not in st.session_state:
    st.session_state.current_theme = 'Terminal Blue'

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'market'

if 'economic_data' not in st.session_state:
    # Sample economic data
    st.session_state.economic_data = {
        'GDP Growth': '2.1%',
        'Unemployment': '3.8%',
        'Inflation': '3.2%',
        'Fed Rate': '5.25-5.50%',
        'Treasury 10Y': '4.25%'
    }

if 'top_performers' not in st.session_state:
    # Sample top performers
    st.session_state.top_performers = [
        {'ticker': 'NVDA', 'change': 5.2, 'volume': '45.3M'},
        {'ticker': 'AAPL', 'change': 3.1, 'volume': '87.7M'},
        {'ticker': 'AMZN', 'change': 2.7, 'volume': '52.1M'},
        {'ticker': 'MSFT', 'change': 2.4, 'volume': '33.5M'},
        {'ticker': 'GOOGL', 'change': 2.1, 'volume': '28.9M'}
    ]

if 'bottom_performers' not in st.session_state:
    # Sample bottom performers
    st.session_state.bottom_performers = [
        {'ticker': 'XOM', 'change': -2.8, 'volume': '22.3M'},
        {'ticker': 'CVX', 'change': -2.3, 'volume': '18.5M'},
        {'ticker': 'JPM', 'change': -1.9, 'volume': '15.2M'},
        {'ticker': 'PFE', 'change': -1.7, 'volume': '33.5M'},
        {'ticker': 'WMT', 'change': -1.2, 'volume': '12.8M'}
    ]

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
        "^VIX": "VIX",
        "^TNX": "10-Year Treasury",  # Added Treasury yield
        "^FTSE": "FTSE 100",         # Added international indices
        "^N225": "Nikkei 225"
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

def fetch_latest_news(tickers=None, limit=15):  # Increased limit
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
    """Enhanced sentiment analysis to predict outcome based on headline"""
    positive_words = ['rise', 'jump', 'gain', 'surge', 'up', 'high', 'growth', 'profit', 
                     'beat', 'exceed', 'positive', 'bullish', 'rally', 'soar', 'outperform',
                     'upgrade', 'boost', 'strong', 'success', 'win', 'approve', 'breakthrough']
    negative_words = ['fall', 'drop', 'decline', 'down', 'low', 'loss', 'miss', 'below', 
                     'negative', 'bearish', 'plunge', 'sink', 'crash', 'struggle', 'downgrade',
                     'weak', 'cut', 'fail', 'disappoint', 'risk', 'concern', 'investigation']
    
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
        data['SMA200'] = data['Close'].rolling(window=200).mean()
        
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
        
        # Calculate Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
        
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
        
        # Bollinger Bands
        if latest['Close'] > latest['BB_Upper']:
            signals['BB'] = {"signal": "SELL", "message": "Price above upper band (overbought)"}
        elif latest['Close'] < latest['BB_Lower']:
            signals['BB'] = {"signal": "BUY", "message": "Price below lower band (oversold)"}
        else:
            signals['BB'] = {"signal": "HOLD", "message": "Price within bands"}
        
        # Golden/Death Cross (SMA50 vs SMA200)
        if 'SMA50' in data.columns and 'SMA200' in data.columns:
            # Check for recent crossover (last 5 days)
            recent_data = data.tail(5)
            crosses = (recent_data['SMA50'] > recent_data['SMA200']) != (recent_data['SMA50'].shift(1) > recent_data['SMA200'].shift(1))
            
            if crosses.any():
                if latest['SMA50'] > latest['SMA200']:
                    signals['Cross'] = {"signal": "BUY", "message": "Recent Golden Cross (50 > 200)"}
                else:
                    signals['Cross'] = {"signal": "SELL", "message": "Recent Death Cross (50 < 200)"}
            else:
                if latest['SMA50'] > latest['SMA200']:
                    signals['Cross'] = {"signal": "HOLD", "message": "SMA50 above SMA200"}
                else:
                    signals['Cross'] = {"signal": "HOLD", "message": "SMA50 below SMA200"}
        
        # Overall signal (simple majority with weighting)
        signal_weights = {"BUY": 0, "HOLD": 0, "SELL": 0}
        
        for indicator, signal_info in signals.items():
            signal_weights[signal_info['signal']] += 1
        
        if signal_weights["BUY"] > signal_weights["SELL"]:
            overall = "BUY"
        elif signal_weights["SELL"] > signal_weights["BUY"]:
            overall = "SELL"
        else:
            overall = "HOLD"
        
        # Add overall signal and current values
        signals['OVERALL'] = {"signal": overall, "message": f"{signal_weights['BUY']} buy vs {signal_weights['SELL']} sell vs {signal_weights['HOLD']} hold"}
        signals['values'] = {
            'price': latest['Close'],
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'signal': latest['Signal'],
            'sma20': latest['SMA20'],
            'sma50': latest['SMA50'],
            'bb_upper': latest['BB_Upper'],
            'bb_lower': latest['BB_Lower']
        }
        
        return signals
    except Exception as e:
        return {"error": str(e)}
