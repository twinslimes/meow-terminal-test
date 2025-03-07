import requests
import re
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import os

app = Flask(__name__)

# API key (consider using environment variable in production)
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
        return "⟷ Neutral", "amber"

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
            return []
    
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

@app.route('/')
def index():
    """Render the main dashboard page"""
    market_events = get_market_events()
    return render_template('index.html', 
                           market_events=market_events)

@app.route('/news')
def news():
    """Fetch and return stock news as JSON"""
    news_items = fetch_stock_news()
    return jsonify(news_items)

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Create HTML template
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Stock News Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f0f0;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        #market-calendar {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .day-column {
            flex: 1;
            border: 1px solid #ddd;
            padding: 10px;
            margin: 0 5px;
            background-color: white;
        }
        .day-header {
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }
        .event {
            margin: 5px 0;
            font-size: 0.9em;
        }
        #news-section {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
        }
        .news-item {
            border-bottom: 1px solid #eee;
            padding: 10px 0;
        }
        .news-item:last-child {
            border-bottom: none;
        }
        .news-headline {
            cursor: pointer;
            color: #1a73e8;
        }
        .news-headline:hover {
            text-decoration: underline;
        }
        #refresh-btn {
            display: block;
            margin: 20px auto;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1>Stock News Dashboard</h1>
    
    <div id="market-calendar">
        {% for day in market_events %}
        <div class="day-column">
            <div class="day-header">{{ day.day }} ({{ day.date }})</div>
            {% for event in day.events %}
            <div class="event" style="color: {{ event.color }};">{{ event.text }}</div>
            {% endfor %}
        </div>
        {% endfor %}
    </div>
    
    <button id="refresh-btn">Refresh News</button>
    
    <div id="news-section">
        <h2>Latest Stock News</h2>
        <div id="news-container"></div>
    </div>

    <script>
        function fetchNews() {
            fetch('/news')
                .then(response => response.json())
                .then(news => {
                    const container = document.getElementById('news-container');
                    container.innerHTML = ''; // Clear previous news
                    
                    if (news.length === 0) {
                        container.innerHTML = '<p>No news available</p>';
                        return;
                    }
                    
                    news.forEach(item => {
                        const newsItem = document.createElement('div');
                        newsItem.className = 'news-item';
                        
                        newsItem.innerHTML = `
                            <div>Tickers: ${item.tickers_str}</div>
                            <div class="news-headline" onclick="window.open('${item.article_url}', '_blank')">
                                ${item.title}
                            </div>
                            <div style="color: ${item.outcome_color};">${item.outcome_text}</div>
                        `;
                        
                        container.appendChild(newsItem);
                    });
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }

        // Fetch news on page load
        fetchNews();

        // Add event listener to refresh button
        document.getElementById('refresh-btn').addEventListener('click', fetchNews);
    </script>
</body>
</html>
        ''')
    
    # Run the Flask app
    app.run(debug=True)
