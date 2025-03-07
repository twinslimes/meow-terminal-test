import streamlit as st
import requests
import re
from datetime import datetime, timedelta

# Constants
API_KEY = "9skphQ6G7_rESW6iTNJDIAycT9gncpje"

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

def get_market_events():
    """Generate market events for the current week"""
    # Get the current week's Monday date
    today = datetime.now().date()
    monday = today - timedelta(days=today.weekday())
    
    events = [
        # Monday events
        {"day": 0, "text": "AAPL Earnings (After Close)", "color": "green"},
        {"day": 0, "text": "Market Open 9:30 AM", "color": "black"},
        
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
    formatted_events = []
    
    for i, day in enumerate(weekdays):
        current_date = monday + timedelta(days=i)
        day_events = [event for event in events if event['day'] == i]
        formatted_events.append({
            "day": day,
            "date": current_date.strftime('%m/%d'),
            "events": day_events
        })
    
    return formatted_events

def fetch_stock_news():
    """Fetch stock news from Polygon API"""
    # Get yesterday's date
    yesterday = datetime.now() - timedelta(days=1)
    date_from = yesterday.strftime("%Y-%m-%d")
    
    # Polygon API endpoint for market news
    url = f"https://api.polygon.io/v2/reference/news?limit=10&order=desc&sort=published_utc&apiKey={API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200 and 'results' in data:
            # Filter for major news (those with tickers mentioned)
            major_news = [item for item in data['results'] if item.get('tickers') and len(item.get('tickers', [])) > 0]
            
            # Enrich news with sentiment prediction
            for news in major_news:
                outcome_text, outcome_color = predict_outcome(news.get('title', ''))
                news['outcome_text'] = outcome_text
                news['outcome_color'] = outcome_color
                news['tickers_str'] = ", ".join(news.get('tickers', []))
            
            return major_news
        else:
            st.error(f"Error fetching news: {data.get('error', 'Unknown error')}")
            return []
    
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def news_dashboard():
    """Streamlit News Dashboard Page"""
    st.title("Stock News Dashboard")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .market-calendar {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .day-column {
        flex: 1;
        border: 1px solid #e0e0e0;
        padding: 10px;
        margin: 0 5px;
        border-radius: 5px;
    }
    .day-header {
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .event {
        margin: 5px 0;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Market Calendar Section
    st.subheader("Market Schedule")
    
    # Get market events
    market_events = get_market_events()
    
    # Create columns for market calendar
    calendar_cols = st.columns(5)
    
    # Display market events
    for i, day in enumerate(market_events):
        with calendar_cols[i]:
            st.markdown(f"**{day['day']} ({day['date']})**")
            for event in day['events']:
                st.markdown(f"<div style='color:{event['color']};'>{event['text']}</div>", unsafe_allow_html=True)
    
    # News Section
    st.subheader("Latest Stock News")
    
    # Fetch and display news
    news_items = fetch_stock_news()
    
    # Display each news item
    for news in news_items:
        st.markdown(f"**Tickers:** {news['tickers_str']}")
        
        # Clickable headline
        st.markdown(f"[{news['title']}]({news['article_url']})")
        
        # Sentiment outcome
        st.markdown(f"**Sentiment:** <span style='color:{news['outcome_color']};'>{news['outcome_text']}</span>", 
                    unsafe_allow_html=True)
        
        # Separator
        st.markdown("---")

# If this script is run directly (not imported)
if __name__ == "__main__":
    news_dashboard()
