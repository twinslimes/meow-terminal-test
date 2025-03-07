import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import re
import json
import time
from datetime import datetime, timedelta
import yfinance as yf

# Set page config
st.set_page_config(
    page_title="Meow Dashboard",
    page_icon="😺",
    layout="wide",
)

# Apply core terminal-style CSS (simplified)
terminal_css = """
<style>
    /* Main terminal theme */
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'VT323', monospace;
        color: #ffffff;
    }
    
    /* Background and main container */
    .main {
        background-color: #2f2f2f;
    }
    
    /* Buttons and inputs */
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
    
    /* Dashboard widget */
    .dashboard-widget {
        border: 1px solid #4a90e2;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #2f2f2f;
    }
    
    /* Widget header */
    .widget-header {
        border-bottom: 1px solid #4a90e2;
        margin-bottom: 10px;
        padding-bottom: 5px;
        color: #e6f3ff;
        font-weight: bold;
    }
    
    /* Stock price colors */
    .stock-up {
        color: #4CAF50 !important;
    }
    
    .stock-down {
        color: #F44336 !important;
    }
    
    /* Live indicator */
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
    
    /* Stock card */
    .stock-card {
        border: 1px solid #4a90e2;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #2f2f2f;
    }
    
    /* News ticker */
    .news-ticker {
        white-space: nowrap;
        overflow: hidden;
        box-sizing: border-box;
        border: 1px solid #4a90e2;
        padding: 8px 10px;
        margin-bottom: 10px;
        position: relative;
    }
    
    .ticker-content {
        display: inline-block;
        animation: ticker 30s linear infinite;
    }
    
    @keyframes ticker {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    
    /* News item */
    .news-item {
        padding: 8px;
        margin-bottom: 8px;
        border: 1px solid #4a90e2;
        border-radius: 5px;
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
</style>
"""

# Initialize session state
def init_session_state():
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []  # Start with empty watchlist
        
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
        
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
        
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = {}
        
    if 'news_data' not in st.session_state:
        st.session_state.news_data = []
        
    if 'refresh_rate' not in st.session_state:
        st.session_state.refresh_rate = 5  # Default refresh rate in seconds

# Functions to save and load watchlist data
def save_watchlist_data():
    """Save watchlist data to browser local storage"""
    data_to_save = {
        'watchlist': st.session_state.watchlist,
        'refresh_rate': st.session_state.refresh_rate
    }
    
    # Serialize to JSON
    json_data = json.dumps(data_to_save)
    
    # Create a JavaScript function to save to localStorage
    st.markdown(f"""
    <script>
        try {{
            localStorage.setItem('meowDashboardData', '{json_data}');
            console.log('Watchlist data saved to localStorage');
            
            // Also save to cookies as backup
            let expiryDate = new Date();
            expiryDate.setTime(expiryDate.getTime() + (365 * 24 * 60 * 60 * 1000));
            document.cookie = "meowDashboardData={json_data}; expires=" + expiryDate.toUTCString() + "; path=/; SameSite=Strict";
        }} catch (error) {{
            console.error('Error saving to localStorage:', error);
        }}
    </script>
    """, unsafe_allow_html=True)

def load_saved_data():
    """Load watchlist data from browser local storage"""
    # Check if we've already set up the retrieval mechanism
    if 'data_retrieval_set' not in st.session_state:
        st.session_state.data_retrieval_set = True
        
        # Inject JavaScript to retrieve data from localStorage and pass it to a callback
        st.markdown("""
        <script>
            try {
                // Try localStorage first
                let savedData = localStorage.getItem('meowDashboardData');
                
                // Fall back to cookies if needed
                if (!savedData) {
                    function getCookie(name) {
                        const value = `; ${document.cookie}`;
                        const parts = value.split(`; ${name}=`);
                        if (parts.length === 2) return parts.pop().split(';').shift();
                        return null;
                    }
                    savedData = getCookie('meowDashboardData');
                }
                
                if (savedData) {
                    // Send saved data to Streamlit via query parameters
                    const dataToSend = encodeURIComponent(savedData);
                    const url = new URL(window.location.href);
                    url.searchParams.set('saved_data', dataToSend);
                    
                    // Update URL without reloading
                    window.history.replaceState(null, '', url.toString());
                    
                    // Force a reload to apply the query params
                    window.location.reload();
                }
            } catch (error) {
                console.error('Error loading saved data:', error);
            }
        </script>
        """, unsafe_allow_html=True)
    
    # Try to get data from URL parameters
    query_params = st.experimental_get_query_params()
    if 'saved_data' in query_params:
        try:
            saved_data_json = query_params['saved_data'][0]
            saved_data = json.loads(saved_data_json)
            
            # Update session state with saved data
            if 'watchlist' in saved_data:
                st.session_state.watchlist = saved_data['watchlist']
            if 'refresh_rate' in saved_data:
                st.session_state.refresh_rate = saved_data['refresh_rate']
                
            # Clear the query params to avoid loading the same data multiple times
            st.experimental_set_query_params()
        except Exception as e:
            st.error(f"Error loading saved data: {e}")

# Data fetching functions
def fetch_stock_data(ticker_list):
    """Fetch current stock data for watchlist"""
    if not ticker_list:
        return {}  # Return empty dict if no tickers
        
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
                    volume = tickers_data['Volume'].iloc[-1]
                else:
                    # Multiple tickers case
                    current_price = tickers_data[ticker]['Close'].iloc[-1]
                    prev_price = tickers_data[ticker]['Close'].iloc[-2]
                    volume = tickers_data[ticker]['Volume'].iloc[-1]
                
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                
                # Fetch additional info about the company
                try:
                    ticker_info = yf.Ticker(ticker).info
                    company_name = ticker_info.get('shortName', 'Unknown')
                    sector = ticker_info.get('sector', 'Unknown')
                    market_cap = ticker_info.get('marketCap', 0)
                    market_cap_display = f"${market_cap/1000000000:.2f}B" if market_cap >= 1000000000 else f"${market_cap/1000000:.2f}M"
                except:
                    company_name = "Unknown"
                    sector = "Unknown"
                    market_cap_display = "Unknown"
                
                result[ticker] = {
                    'price': current_price,
                    'change': change,
                    'change_pct': change_pct,
                    'volume': volume,
                    'volume_display': f"{volume/1000000:.1f}M" if volume >= 1000000 else f"{volume/1000:.1f}K",
                    'company_name': company_name,
                    'sector': sector,
                    'market_cap': market_cap_display,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
            except Exception as e:
                result[ticker] = {
                    'price': 0.0,
                    'change': 0.0,
                    'change_pct': 0.0,
                    'error': str(e),
                    'company_name': 'Error',
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
    except Exception as general_e:
        st.error(f"Error fetching stock data: {str(general_e)}")
    
    return result

def fetch_latest_news(tickers=None, limit=10):
    """Fetch latest stock market news"""
    try:
        news_list = []
        
        # If no specific tickers provided, use a general market news source
        if not tickers or len(tickers) == 0:
            # Fallback to some major stock symbols for news
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
        
        # Loop through tickers to get news
        for ticker in tickers[:5]:  # Limit to first 5 tickers to avoid too many requests
            try:
                # Get news from Yahoo Finance
                ticker_obj = yf.Ticker(ticker)
                ticker_news = ticker_obj.news
                
                for news in ticker_news[:3]:  # Get top 3 news per ticker
                    # Format the timestamp
                    news_date = datetime.fromtimestamp(news.get('providerPublishTime', 0))
                    formatted_date = news_date.strftime("%m-%d %H:%M")
                    
                    # Simple sentiment analysis based on title
                    outcome_text, outcome_class = predict_news_outcome(news.get('title', ''))
                    
                    news_list.append({
                        'ticker': ticker,
                        'title': news.get('title', 'No headline available'),
                        'url': news.get('link', '#'),
                        'source': news.get('publisher', 'Yahoo Finance'),
                        'date': formatted_date,
                        'sentiment': outcome_text,
                        'sentiment_class': outcome_class
                    })
            except Exception as e:
                # Just skip this ticker if there's an error
                pass
        
        # Sort by date (newest first) and limit
        news_list = sorted(news_list, key=lambda x: x['date'], reverse=True)
        return news_list[:limit]
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

def predict_news_outcome(title):
    """Simple sentiment analysis to predict outcome based on headline"""
    positive_words = ['rise', 'jump', 'gain', 'surge', 'up', 'high', 'growth', 'profit', 
                     'beat', 'exceed', 'positive', 'bullish', 'rally', 'soar', 'outperform']
    negative_words = ['fall', 'drop', 'decline', 'down', 'low', 'loss', 'miss', 'below', 
                     'negative', 'bearish', 'plunge', 'sink', 'crash', 'struggle', 'downgrade']
    
    title_lower = title.lower()
    
    positive_count = sum(1 for word in positive_words if re.search(r'\b' + word + r'\b', title_lower))
    negative_count = sum(1 for word in negative_words if re.search(r'\b' + word + r'\b', title_lower))
    
    if positive_count > negative_count:
        return "📈 Positive", "news-positive"
    elif negative_count > positive_count:
        return "📉 Negative", "news-negative"
    else:
        return "⟷ Neutral", "news-neutral"

def fetch_intraday_data(ticker):
    """Fetch intraday data for stock chart"""
    try:
        # Get intraday data for the chart
        data = yf.download(
            tickers=ticker,
            period="1d",
            interval="5m",
            auto_adjust=True,
            prepost=True
        )
        
        # Reset index to have datetime as a column
        data = data.reset_index()
        return data
    except Exception as e:
        st.error(f"Error fetching intraday data: {str(e)}")
        return pd.DataFrame()

# Function to refresh all data
def refresh_all_data():
    """Refresh all dashboard data"""
    st.session_state.refresh_counter += 1
    st.session_state.last_refresh = datetime.now()
    
    # Fetch stock data for watchlist
    if st.session_state.watchlist:
        st.session_state.stock_data = fetch_stock_data(st.session_state.watchlist)
    
    # Fetch news
    st.session_state.news_data = fetch_latest_news(st.session_state.watchlist)

# Display functions
def display_watchlist_manager():
    """Display watchlist management widget"""
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header">Watchlist Manager</div>', unsafe_allow_html=True)
    
    # Add a stock to watchlist
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_ticker = st.text_input("Add ticker symbol:", key="add_ticker").upper()
    with col2:
        add_button = st.button("Add", key="add_button")
        
    if add_button and new_ticker:
        if new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
            # Save watchlist after adding
            save_watchlist_data()
            # Refresh data to include new ticker
            refresh_all_data()
            st.rerun()
    
    # Show current watchlist with remove buttons
    if not st.session_state.watchlist:
        st.markdown("<p>Your watchlist is empty. Add ticker symbols above to start tracking stocks.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p>Your current watchlist:</p>", unsafe_allow_html=True)
        
        for i, ticker in enumerate(st.session_state.watchlist):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"<div>{ticker}</div>", unsafe_allow_html=True)
            with col2:
                # Use a unique key for each remove button
                if st.button("🗑️", key=f"remove_{ticker}_{i}"):
                    st.session_state.watchlist.remove(ticker)
                    # Save watchlist after removing
                    save_watchlist_data()
                    st.rerun()
    
    # Clear watchlist button
    if st.session_state.watchlist and st.button("Clear All"):
        st.session_state.watchlist = []
        save_watchlist_data()
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_live_watchlist():
    """Display live watchlist with stock cards"""
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header"><span class="live-indicator"></span>Live Watchlist</div>', unsafe_allow_html=True)
    
    # No stocks message
    if not st.session_state.watchlist:
        st.markdown("<p>Add stocks to your watchlist to see live data.</p>", unsafe_allow_html=True)
    else:
        # No data message
        if not st.session_state.stock_data:
            st.markdown("<p>Loading stock data...</p>", unsafe_allow_html=True)
        else:
            # Display stocks in a grid
            cols = st.columns(3)  # Create 3 columns for the grid
            
            for i, ticker in enumerate(st.session_state.watchlist):
                col_idx = i % 3  # Determine which column to place the stock card
                
                with cols[col_idx]:
                    if ticker in st.session_state.stock_data:
                        data = st.session_state.stock_data[ticker]
                        
                        if 'error' in data:
                            # Error card
                            st.markdown(f"""
                            <div class="stock-card">
                                <h3>{ticker}</h3>
                                <p>Error loading data: {data['error']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Determine color based on price change
                            color_class = "stock-up" if data['change'] >= 0 else "stock-down"
                            change_sign = "+" if data['change'] >= 0 else ""
                            
                            # Stock card with live data
                            st.markdown(f"""
                            <div class="stock-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <h3>{ticker}</h3>
                                    <div style="font-size: 12px; color: #e6f3ff;">Updated: {data['timestamp']}</div>
                                </div>
                                <p>{data['company_name']}</p>
                                <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                                    <div style="font-size: 24px;">${data['price']:.2f}</div>
                                    <div class="{color_class}" style="font-size: 18px;">{change_sign}{data['change_pct']:.2f}%</div>
                                </div>
                                <div style="display: grid; grid-template-columns: 1fr 1fr; grid-gap: 5px; font-size: 14px;">
                                    <div>Sector: {data['sector']}</div>
                                    <div>Mkt Cap: {data['market_cap']}</div>
                                    <div>Change: {change_sign}${data['change']:.2f}</div>
                                    <div>Volume: {data['volume_display']}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display intraday chart for the stock
                            intraday_data = fetch_intraday_data(ticker)
                            if not intraday_data.empty:
                                fig = px.line(
                                    intraday_data, 
                                    x='Datetime', 
                                    y='Close',
                                    title=f"{ticker} Today"
                                )
                                
                                # Update layout for terminal theme
                                fig.update_layout(
                                    height=200,
                                    margin=dict(l=0, r=0, t=30, b=0),
                                    paper_bgcolor='#2f2f2f',
                                    plot_bgcolor='#2f2f2f',
                                    font=dict(color='#ffffff'),
                                    xaxis=dict(gridcolor='#4a90e2'),
                                    yaxis=dict(gridcolor='#4a90e2')
                                )
                                
                                fig.update_traces(
                                    line_color='#4a90e2' if data['change'] >= 0 else '#F44336'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Loading card
                        st.markdown(f"""
                        <div class="stock-card">
                            <h3>{ticker}</h3>
                            <p>Loading data...</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_news_widget():
    """Display news widget with ticker"""
    st.markdown('<div class="dashboard-widget">', unsafe_allow_html=True)
    st.markdown('<div class="widget-header"><span class="live-indicator"></span>Market News</div>', unsafe_allow_html=True)
    
    # News ticker for latest headlines
    if st.session_state.news_data:
        # Get the 5 most recent headlines
        recent_headlines = [f"{news['ticker']}: {news['title']}" for news in st.session_state.news_data[:5]]
        headlines_str = " ★ ".join(recent_headlines)
        
        st.markdown(f"""
        <div class="news-ticker">
            <div class="ticker-content">
                {headlines_str} ★ {headlines_str}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display news list
    if not st.session_state.news_data:
        st.markdown("<p>Loading news...</p>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="scrollable">', unsafe_allow_html=True)
        
        for news in st.session_state.news_data:
            # Display news item
            st.markdown(f"""
            <div class="news-item">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: bold;">{news['ticker']}</span>
                    <span>{news['date']}</span>
                </div>
                <a href="{news['url']}" target="_blank" style="color: #e6f3ff; text-decoration: none;">{news['title']}</a>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span>{news['source']}</span>
                    <span class="{news['sentiment_class']}">{news['sentiment']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_settings():
    """Display dashboard settings sidebar"""
    st.sidebar.markdown("<h3 style='color: #e6f3ff;'>Dashboard Settings</h3>", unsafe_allow_html=True)
    
    # Refresh rate slider
    refresh_rate = st.sidebar.slider(
        "Refresh interval (seconds)",
        min_value=1,
        max_value=60,
        value=st.session_state.refresh_rate
    )
    
    if refresh_rate != st.session_state.refresh_rate:
        st.session_state.refresh_rate = refresh_rate
        save_watchlist_data()
    
    # Enable/disable auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=True)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        refresh_all_data()
        st.rerun()
    
    # Force save button
    if st.sidebar.button("💾 Save Watchlist"):
        save_watchlist_data()
        st.sidebar.success("Watchlist saved!")
    
    # Help information
    with st.sidebar.expander("Help"):
        st.markdown("""
        ### How to use Meow Dashboard
        
        1. **Add stocks** to your watchlist using ticker symbols (e.g., AAPL, MSFT)
        2. **Remove stocks** by clicking the 🗑️ button
        3. **Live data** refreshes automatically based on your refresh interval
        4. **Save** your watchlist to keep it when you return
        
        Your watchlist is saved in your browser and will be available when you come back!
        """)
    
    return auto_refresh

def main():
    """Main function to run the application"""
    try:
        # Apply CSS
        st.markdown(terminal_css, unsafe_allow_html=True)
        
        # Initialize session state
        init_session_state()
        
        # Load saved data
        load_saved_data()
        
        # Title
        st.markdown("<h1 style='text-align: center; color: #e6f3ff;'>🐱 Meow Live Watchlist 🐱</h1>", unsafe_allow_html=True)
        
        # Display settings sidebar
        auto_refresh = display_settings()
        
        # Initial data fetch if empty
        if not st.session_state.stock_data and st.session_state.watchlist:
            refresh_all_data()
        
        # Auto-refresh logic
        if auto_refresh:
            # Check if refresh_rate seconds have passed since last refresh
            time_diff = (datetime.now() - st.session_state.last_refresh).total_seconds()
            if time_diff >= st.session_state.refresh_rate:
                refresh_all_data()
                st.rerun()
            
            # Display refresh indicator
            seconds_to_refresh = max(0, st.session_state.refresh_rate - int(time_diff))
            st.markdown(f"""
            <div class="refresh-indicator">
                Auto-refresh: {seconds_to_refresh}s | Last: {st.session_state.last_refresh.strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
        
        # Layout - use tabs for better organization
        tab1, tab2 = st.tabs(["Watchlist", "News"])
        
        with tab1:
            # Watchlist tab
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Watchlist manager
                display_watchlist_manager()
            
            with col2:
                # Live watchlist
                display_live_watchlist()
        
        with tab2:
            # News tab
            display_news_widget()
        
        # Footer
        st.markdown("""
        <div style="margin-top: 30px; text-align: center; color: #e6f3ff; font-size: 12px;">
            <p>© 2025 Meow Terminal | Simple Live Watchlist</p>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        # Error handling
        st.error(f"An error occurred: {str(e)}")
        
        # Recovery options
        st.write("### Recovery Options")
        
        if st.button("Reset Dashboard"):
            # Clear session state to recover from corrupted state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# This is the entry point of the script
if __name__ == "__main__":
    main()
