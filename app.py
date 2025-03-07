import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import subprocess
import threading
import sys
import os
import requests
import re
from datetime import datetime, timedelta

# Import local modules
from data_utils import get_api_keys, fetch_additional_stock_data, calculate_technical_indicators
from visualization import create_price_distribution_plot, create_model_comparison_plot, create_confidence_interval_plot
from analysis import display_stock_analysis_section, display_technical_indicators_section, display_fundamental_analysis_section
from day_trader import display_day_trader_section
from backtesting import display_backtesting_section

# Import all model classes from models.py
from models import (
    ModelType, StockData, StockPriceModel, GeometricBrownianMotion, 
    AdvancedGBM, JumpDiffusionModel, HestonModel, GARCHModel,
    RegimeSwitchingModel, QuasiMonteCarloModel, VarianceGammaModel, 
    NeuralSDEModel, StockModelEnsemble, HAS_ARCH
)

# Suppress warnings in the UI
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Meow Terminal",
    page_icon="😺",
    layout="wide",
)

# Apply terminal-style CSS with black, blue, gray, and muted white-blue accents
terminal_css = """
<style>
    /* Main terminal theme */
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'VT323', monospace;
        color: #ffffff; /* White text for readability on dark backgrounds */
        caret-color: #4a90e2; /* Muted blue cursor for contrast */
    }
    
    /* Background and main container */
    .main {
        background-color: #2f2f2f; /* Dark gray background for uniformity */
        background-image: linear-gradient(rgba(74, 144, 226, 0.05) 50%, transparent 50%);
        background-size: 100% 4px;
    }
    
    /* Old Windows style border (muted white-blue) */
    div.stButton > button, .stSelectbox > div > div, div.stNumberInput > div > div {
        border: 2px solid #e6f3ff !important; /* Muted white-blue border */
        border-right: 2px solid #000 !important;
        border-bottom: 2px solid #000 !important;
        background-color: #4a90e2 !important; /* Muted blue for buttons */
        color: #ffffff !important; /* White text for contrast */
        font-family: 'VT323', monospace !important;
        font-size: 18px !important;
    }
    
    div.stButton > button:active {
        border: 2px solid #000 !important;
        border-right: 2px solid #e6f3ff !important;
        border-bottom: 2px solid #e6f3ff !important;
    }
    
    /* Text inputs */
    div.stTextInput > div > div > input {
        background-color: #2f2f2f; /* Dark gray background for inputs */
        color: #ffffff; /* White text */
        border: 1px solid #e6f3ff; /* Muted white-blue border */
        font-family: 'VT323', monospace !important;
        font-size: 18px !important;
    }
    
    /* Metrics */
    div.stMetric > div {
        background-color: #2f2f2f; /* Dark gray background for metrics */
        border: 1px solid #e6f3ff; /* Muted white-blue border */
        padding: 10px;
    }
    
    div.stMetric label {
        color: #e6f3ff !important; /* Muted white-blue labels */
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e6f3ff !important; /* Muted white-blue for headers */
        font-family: 'VT323', monospace !important;
    }
    
    /* Sidebar (match main background for uniformity) */
    section[data-testid="stSidebar"] {
        background-color: #2f2f2f; /* Dark gray, matching main background */
        border-right: 2px solid #e6f3ff; /* Muted white-blue border */
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        color: #e6f3ff !important; /* Muted white-blue text */
    }
    
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] button {
        font-family: 'VT323', monospace !important;
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        background-color: #2f2f2f !important; /* Dark gray background for charts */
    }
    
    /* Slider handle */
    .stSlider > div > div > div > div {
        background-color: #e6f3ff !important; /* Muted white-blue slider handle */
    }
    
    /* CRT effect overlay (muted blue-gray) */
    .main::before {
        content: " ";
        display: block;
        position: fixed;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(rgba(74, 144, 226, 0) 50%, rgba(47, 47, 47, 0.15) 50%), 
                    linear-gradient(90deg, rgba(74, 144, 226, 0.05), rgba(47, 47, 47, 0.02), rgba(74, 144, 226, 0.05));
        z-index: 999;
        background-size: 100% 2px, 3px 100%;
        pointer-events: none;
    }
    
    /* Terminal-style inputs and boxes */
    div.stTextInput, div.stNumberInput {
        background-color: #2f2f2f; /* Dark gray background for inputs */
    }
    
    /* Tables with terminal styling */
    div.stTable, div.dataframe {
        background-color: #2f2f2f !important; /* Dark gray background for tables */
        color: #ffffff !important; /* White text */
        border: 1px solid #e6f3ff !important; /* Muted white-blue border */
    }
    
    /* Tabs */
    button[role="tab"] {
        background-color: #2f2f2f !important; /* Dark gray background for tabs */
        color: #e6f3ff !important; /* Muted white-blue text */
        border: 1px solid #e6f3ff !important; /* Muted white-blue border */
    }
    
    button[role="tab"][aria-selected="true"] {
        background-color: #4a90e2 !important; /* Muted blue for selected tab */
        border-bottom: 2px solid #e6f3ff !important; /* Muted white-blue border */
    }
    
    div[role="tabpanel"] {
        background-color: #2f2f2f !important; /* Dark gray background for tab panels */
        border: 1px solid #e6f3ff !important; /* Muted white-blue border */
    }
    
    /* Success/info messages */
    div.stSuccessMessage, div.stInfoMessage {
        background-color: #4a90e2 !important; /* Muted blue for messages */
        color: #ffffff !important; /* White text */
    }
    
    /* Windows 95-style title bar for sections */
    .win95-header {
        background-color: #4a90e2; /* Muted blue for headers */
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
        background-color: #2f2f2f; /* Dark gray for panels */
        border-top: 2px solid #e6f3ff;
        border-left: 2px solid #e6f3ff;
        border-right: 2px solid #000000;
        border-bottom: 2px solid #000000;
        padding: 5px;
        margin: 10px 0;
    }
    
    /* Expander styling */
    details {
        background-color: #2f2f2f !important; /* Dark gray background for expanders */
        border: 1px solid #e6f3ff !important; /* Muted white-blue border */
    }
    
    details summary {
        color: #e6f3ff !important; /* Muted white-blue text */
        font-family: 'VT323', monospace !important;
    }
    
    /* Special terminal blinking cursor (muted white-blue) */
    .terminal-cursor::after {
        content: "▌";
        animation: blink 1s step-end infinite;
        font-weight: bold;
        color: #e6f3ff; /* Muted white-blue cursor */
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    
    /* Fix for checkbox color */
    .stCheckbox label p {
        color: #e6f3ff !important; /* Muted white-blue for checkboxes */
    }
    
    /* News item styling */
    .news-item {
        border: 1px solid #4a90e2;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        background-color: #2f2f2f;
    }
    
    .news-sentiment {
        border: 1px solid;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
    }
</style>
"""

# Function to create a Windows 95-style header
def win95_header(text):
    return f'<div class="win95-header">{text}</div>'

def display_homepage():
    """Display the homepage of the application"""
    st.markdown(terminal_css, unsafe_allow_html=True)
    
    # Title with terminal styling
    st.markdown("<h1 style='color: #e6f3ff; text-align: center;'>🐱 Meow Terminal 🐱</h1>", unsafe_allow_html=True)
    
    # Button to enter app
    if st.button("Enter Terminal", key="enter_app", use_container_width=True):
        st.session_state.show_dashboard = True
        st.rerun()
    
    # YouTube Video below button
    st.video("https://www.youtube.com/watch?v=l9QTwRn_vmc&t=1s&ab_channel=twinslimes")  # User's actual video
    
    # Footer
    st.markdown("""
    <div style="margin-top: 20px; text-align: center; color: #e6f3ff; font-size: 12px;">
        <p>© 2025 Meow Terminal</p>
    </div>
    """, unsafe_allow_html=True)

# News dashboard functionality
def predict_outcome(title):
    """Simple sentiment analysis to predict outcome based on headline"""
    positive_words = ['rise', 'jump', 'gain', 'surge', 'up', 'high', 'growth', 'profit', 
                     'beat', 'exceed', 'positive', 'bullish', 'rally', 'soar']
    negative_words = ['fall', 'drop', 'decline', 'down', 'low', 'loss', 'miss', 'below', 
                     'negative', 'bearish', 'plunge', 'sink', 'crash', 'struggle']
    
    title_lower = title.lower()
    
    positive_count = sum(1 for word in positive_words if re.search(r'\b' + word + r'\b', title_lower))
    negative_count = sum(1 for word in negative_words if re.search(r'\b' + word + r'\b', title_lower))
    
    if positive_count > negative_count:
        return "📈 Positive", "green"
    elif negative_count > positive_count:
        return "📉 Negative", "red"
    else:
        return "⟷ Neutral", "orange"

def fetch_ticker_news(ticker):
    """Fetch news specifically for the selected ticker"""
    API_KEY = "9skphQ6G7_rESW6iTNJDIAycT9gncpje"
    # Polygon API endpoint for ticker-specific news
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=10&order=desc&sort=published_utc&apiKey={API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200 and 'results' in data:
            # Process news items
            for news in data['results']:
                outcome_text, outcome_color = predict_outcome(news.get('title', ''))
                news['outcome_text'] = outcome_text
                news['outcome_color'] = outcome_color
                news['tickers_str'] = ", ".join(news.get('tickers', [ticker]))
            
            return data['results']
        else:
            st.warning(f"Error fetching news for {ticker}: {data.get('error', 'Unknown error')}")
            return []
    
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def fetch_market_news():
    """Fetch general market news"""
    API_KEY = "9skphQ6G7_rESW6iTNJDIAycT9gncpje"
    # Polygon API endpoint for market news
    url = f"https://api.polygon.io/v2/reference/news?limit=10&order=desc&sort=published_utc&apiKey={API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200 and 'results' in data:
            # Filter for major news (those with tickers mentioned)
            major_news = [item for item in data['results'] if item.get('tickers') and len(item.get('tickers', [])) > 0]
            
            # Process news items
            for news in major_news:
                outcome_text, outcome_color = predict_outcome(news.get('title', ''))
                news['outcome_text'] = outcome_text
                news['outcome_color'] = outcome_color
                news['tickers_str'] = ", ".join(news.get('tickers', []))
            
            return major_news
        else:
            st.warning(f"Error fetching market news: {data.get('error', 'Unknown error')}")
            return []
    
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def display_news_dashboard_section(ticker):
    """Display news dashboard directly integrated into the app."""
    st.header("Stock News Dashboard")
    
    # Market Calendar Section
    st.subheader("Market Schedule")
    
    # Get the current week's Monday date
    today = datetime.now().date()
    monday = today - timedelta(days=today.weekday())
    
    # Define market events
    events = [
        # Monday events
        {"day": 0, "text": "AAPL Earnings (After Close)", "color": "green"},
        {"day": 0, "text": "Market Open 9:30 AM", "color": "#e6f3ff"},
        
        # Tuesday events
        {"day": 1, "text": "CPI Data 8:30 AM", "color": "orange"},
        {"day": 1, "text": "MSFT Earnings Call", "color": "green"},
        
        # Wednesday events
        {"day": 2, "text": "FOMC Meeting", "color": "red"},
        {"day": 2, "text": "TSLA Earnings", "color": "green"},
        {"day": 2, "text": "Oil Inventory 10:30 AM", "color": "orange"},
        
        # Thursday events
        {"day": 3, "text": "Jobless Claims 8:30 AM", "color": "orange"},
        {"day": 3, "text": "AMZN Earnings (After Close)", "color": "green"},
        
        # Friday events
        {"day": 4, "text": "GOOG Earnings", "color": "green"},
        {"day": 4, "text": "PMI Data 9:45 AM", "color": "orange"}
    ]
    
    # Prepare weekday labels with dates
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    market_events = []
    
    for i, day in enumerate(weekdays):
        current_date = monday + timedelta(days=i)
        day_events = [event for event in events if event['day'] == i]
        market_events.append({
            "day": day,
            "date": current_date.strftime('%m/%d'),
            "events": day_events
        })
    
    # Create columns for market calendar
    calendar_cols = st.columns(5)
    
    # Display market events
    for i, day in enumerate(market_events):
        with calendar_cols[i]:
            st.markdown(f"<div style='font-weight: bold; text-align: center; color: #e6f3ff;'>{day['day']} ({day['date']})</div>", unsafe_allow_html=True)
            for event in day['events']:
                st.markdown(f"<div style='color:{event['color']};'>{event['text']}</div>", unsafe_allow_html=True)
    
    # News Section
    st.subheader(f"Latest News for {ticker}")
    
    # Fetch news - first try ticker-specific, then fallback to general
    with st.spinner(f"Fetching news for {ticker}..."):
        news_items = fetch_ticker_news(ticker)
        
        if not news_items:
            st.info(f"No specific news found for {ticker}. Displaying general market news.")
            news_items = fetch_market_news()
        
        if not news_items:
            st.error("Could not fetch any news. Please try again later.")
    
    # Display news items
    for news in news_items:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Clickable headline
            st.markdown(f"#### [{news['title']}]({news.get('article_url', '#')})")
            
            # Source and publication date
            published_date = datetime.strptime(news.get('published_utc', '')[:19], "%Y-%m-%dT%H:%M:%S") if 'published_utc' in news else None
            date_str = published_date.strftime("%Y-%m-%d %H:%M") if published_date else "N/A"
            
            st.markdown(f"**Source:** {news.get('publisher', {}).get('name', 'Unknown')} | **Published:** {date_str}")
            
            # Tickers mentioned
            st.markdown(f"**Tickers:** {news.get('tickers_str', '')}")
        
        with col2:
            # Sentiment analysis
            outcome_text = news.get('outcome_text', '⟷ Neutral')
            outcome_color = news.get('outcome_color', 'orange')
            
            st.markdown(f"""
            <div class="news-sentiment" style="border-color: {outcome_color};">
                <span style="color: {outcome_color};">{outcome_text}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Description snippet if available
        if 'description' in news and news['description']:
            st.markdown(f"_{news['description']}_")
        
        # Separator
        st.markdown("---")
    
    # Add a refresh button
    if st.button("Refresh News", key="refresh_news", use_container_width=True):
        st.experimental_rerun()

def clear_analysis_results():
    """Clear analysis results when stock data changes."""
    # Clear prediction results
    if 'ensemble' in st.session_state:
        del st.session_state.ensemble
    if 'ensemble_result' in st.session_state:
        del st.session_state.ensemble_result
    if 'target_price' in st.session_state:
        del st.session_state.target_price
    if 'T' in st.session_state:
        del st.session_state.T

# ... rest of app.py file remains the same ...

def main():
    """Main function to run the application"""
    # Apply terminal CSS
    st.markdown(terminal_css, unsafe_allow_html=True)
    
    # Initialize session state to track if dashboard should be shown
    if 'show_dashboard' not in st.session_state:
        st.session_state.show_dashboard = False
    
    # Check if we should show homepage or dashboard
    if not st.session_state.show_dashboard:
        display_homepage()
        return
    
    # Debug info - add this to help identify where the issue might be
    st.markdown("<div style='color: #e6f3ff; margin-bottom: 10px;'>DEBUG: Dashboard is loading...</div>", unsafe_allow_html=True)
    
    # Get API keys
    try:
        alpha_vantage_key, fred_api_key = get_api_keys()
        st.markdown("<div style='color: #e6f3ff; margin-bottom: 10px;'>DEBUG: API keys loaded</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error getting API keys: {e}")
        alpha_vantage_key = "demo"
        fred_api_key = "demo"
    
    # Sidebar for navigation
    try:
        st.sidebar.markdown("<h1 style='color: #e6f3ff;'>Navigation</h1>", unsafe_allow_html=True)
        
        # All navigation options in a single dropdown
        selected_section = st.sidebar.selectbox(
            "Go to",
            ["Stock Dashboard", "Day Trader", "Backtesting", "Stock Analysis", "Technical Indicators", "Fundamental Analysis", "News"]
        )
        
        # Return to homepage button
        if st.sidebar.button("Return to Homepage"):
            st.session_state.show_dashboard = False
            st.rerun()
        
        # Sidebar for inputs (common across sections)
        st.sidebar.markdown("<h2 style='color: #e6f3ff;'>Stock Selection</h2>", unsafe_allow_html=True)
        
        # Fake terminal prompt for stock ticker
        st.sidebar.markdown("<span style='color: #e6f3ff;'>C:\\STOCKS\\> Enter ticker:</span>", unsafe_allow_html=True)
        
        # User inputs for stock ticker
        ticker = st.sidebar.text_input("", value="AAPL", label_visibility="collapsed").upper()
        # Store current ticker in session state for verification
        st.session_state.current_ticker = ticker
        
        # Check if ticker has changed and clear analysis if needed
        if 'last_analyzed_ticker' in st.session_state and st.session_state.last_analyzed_ticker != ticker:
            clear_analysis_results()
        
        # Button to fetch data
        if st.sidebar.button("Fetch Stock Data", use_container_width=True):
            with st.spinner("Fetching data - Please wait..."):
                # Clear previous analysis results when fetching new data
                clear_analysis_results()
                
                try:
                    # Initialize stock data for models
                    stock_data = StockData(ticker, alpha_vantage_key, fred_api_key)
                    stock_data.fetch_data()
                    
                    # Fetch additional data for analysis
                    additional_data = fetch_additional_stock_data(ticker)
                    
                    # Calculate technical indicators
                    if additional_data and 'history' in additional_data and not additional_data['history'].empty:
                        technical_indicators = calculate_technical_indicators(additional_data['history'])
                    else:
                        technical_indicators = None
                    
                    # Store in session state for later use
                    st.session_state.stock_data = stock_data
                    st.session_state.additional_data = additional_data
                    st.session_state.technical_indicators = technical_indicators
                    st.session_state.last_analyzed_ticker = ticker
                    
                    st.sidebar.success(f"Data for {ticker} fetched successfully!")
                    # Force refresh to reflect new data
                    st.rerun()
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
        
        # Terminal breadcrumb path at top
        current_path = f"C:\\> STOCKS\\{ticker}\\{selected_section.upper().replace(' ', '_')}"
        st.markdown(f"<div style='color: #e6f3ff; font-family: monospace; margin-bottom: 10px;'>{current_path}</div>", unsafe_allow_html=True)
        
        # Display the appropriate content based on the selection
        try:
            if selected_section == "Stock Dashboard":
                # If we have analysis results for the current ticker, show them
                if ('ensemble_result' in st.session_state and 
                    'analysis_ticker' in st.session_state and 
                    st.session_state.analysis_ticker == ticker):
                    display_prediction_results()
                else:
                    # Otherwise show the basic dashboard
                    display_basic_dashboard(ticker)
                    
            elif selected_section == "Day Trader":
                display_day_trader_section(ticker)
                
            elif selected_section == "Backtesting":
                display_backtesting_section(ticker)
                    
            elif selected_section == "Stock Analysis":
                display_stock_analysis_section(ticker)
                
            elif selected_section == "Technical Indicators":
                display_technical_indicators_section(ticker)
                
            elif selected_section == "Fundamental Analysis":
                display_fundamental_analysis_section(ticker)
                
            # Use the integrated news dashboard
            elif selected_section == "News":
                display_news_dashboard_section(ticker)
                
        except Exception as e:
            st.error(f"Error displaying {selected_section}: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
    
    except Exception as e:
        st.error(f"Error setting up dashboard: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        
        # Fall back to a simple dashboard if everything else fails
        st.title("Meow Terminal")
        st.markdown("**Debug Mode - Dashboard Failed to Load**")
        st.write("There was an error setting up the dashboard components. Please check the error message above.")
        
        # Add a button to return to the homepage
        if st.button("Return to Homepage"):
            st.session_state.show_dashboard = False
            st.rerun()

# This is the entry point of the script
if __name__ == "__main__":
    main()
