def display_news_dashboard_section(ticker):
    """Display news dashboard directly integrated into the app."""
    st.header("Stock News Dashboard")
    
    # Import necessary modules for news dashboard
    import re
    from datetime import datetime, timedelta
    
    # Custom CSS for better styling - matched to terminal theme
    st.markdown("""
    <style>
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
    """, unsafe_allow_html=True)
    
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
    
    # Simple sentiment analysis function
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
    
    # Function to fetch news for the specific ticker
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
    
    # Function to fetch general market news
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
