import requests
import re
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
import webbrowser

class NewsApp:
    def __init__(self, root, api_key):
        self.root = root
        self.api_key = api_key
        self.root.title("Stock News Dashboard")
        self.root.geometry("800x600")
        
        # Configure the main window
        self.root.configure(bg="#f0f0f0")
        
        # Main container frame
        main_container = tk.Frame(root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title at the very top
        title_label = tk.Label(main_container, text="Major Stock News", font=("Helvetica", 24), bg="#f0f0f0")
        title_label.pack(pady=10)
        
        # =====================================================================
        # MARKET CALENDAR SECTION - Prominently displayed at the top
        # =====================================================================
        calendar_frame = tk.LabelFrame(main_container, text="Market Schedule", font=("Helvetica", 14, "bold"), 
                                     bg="#f0f0f0", padx=10, pady=10)
        calendar_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Get the current week's Monday date
        today = datetime.now().date()
        monday = today - timedelta(days=today.weekday())
        
        # Create week label
        week_label = tk.Label(calendar_frame, 
                             text=f"Week of {monday.strftime('%B %d, %Y')}",
                             font=("Helvetica", 12, "bold"),
                             bg="#f0f0f0")
        week_label.pack(pady=5)
        
        # Create weekday headers frame
        weekday_frame = tk.Frame(calendar_frame, bg="#f0f0f0")
        weekday_frame.pack(fill=tk.X, pady=5)
        
        # Create 5 columns (Monday to Friday)
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        day_frames = []
        
        for i, day in enumerate(weekdays):
            current_date = monday + timedelta(days=i)
            # Frame for each day
            day_column = tk.Frame(weekday_frame, bg="#f0f0f0", bd=1, relief=tk.GROOVE)
            day_column.grid(row=0, column=i, padx=2, sticky="nsew")
            
            # Configure column weight for equal sizing
            weekday_frame.columnconfigure(i, weight=1)
            
            # Day header with date
            day_header = tk.Label(day_column, 
                                 text=f"{day}\n{current_date.strftime('%m/%d')}",
                                 font=("Helvetica", 10, "bold"),
                                 bg="#f0f0f0")
            day_header.pack(fill=tk.X, pady=2)
            
            ttk.Separator(day_column, orient='horizontal').pack(fill='x')
            
            # Add to list for adding events later
            day_frames.append(day_column)
        
        # Add sample events to the calendar
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
        
        # Add events to respective day columns
        for event in events:
            day_idx = event["day"]
            if 0 <= day_idx < 5:  # Monday to Friday
                event_label = tk.Label(day_frames[day_idx],
                                     text=event["text"],
                                     fg=event["color"],
                                     bg="#f0f0f0",
                                     font=("Helvetica", 9),
                                     anchor="w")
                event_label.pack(fill=tk.X, padx=2, pady=1, anchor="w")
        
        # =====================================================================
        # NEWS SECTION
        # =====================================================================
        
        # Add a refresh button
        refresh_button = tk.Button(main_container, text="Refresh News", command=self.fetch_news, 
                                  font=("Helvetica", 12), bg="#4CAF50", fg="white")
        refresh_button.pack(pady=10)
        
        # Create a frame for the news items
        self.news_frame = tk.Frame(main_container, bg="#f0f0f0")
        self.news_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initial news fetch
        self.fetch_news()
    
    def open_article(self, url):
        webbrowser.open_new(url)
    
    def predict_outcome(self, title):
        """Simple sentiment analysis to predict outcome based on headline"""
        positive_words = ['rise', 'jump', 'gain', 'surge', 'up', 'high', 'growth', 'profit', 
                         'beat', 'exceed', 'positive', 'bullish', 'rally', 'soar']
        negative_words = ['fall', 'drop', 'decline', 'down', 'low', 'loss', 'miss', 'below', 
                         'negative', 'bearish', 'plunge', 'sink', 'crash', 'struggle']
        
        title_lower = title.lower()
        
        positive_count = sum(1 for word in positive_words if re.search(r'\b' + word + r'\b', title_lower))
        negative_count = sum(1 for word in negative_words if re.search(r'\b' + word + r'\b', title_lower))
        
        if positive_count > negative_count:
            return "📈 Positive", "#4CAF50"  # Green
        elif negative_count > positive_count:
            return "📉 Negative", "#F44336"  # Red
        else:
            return "⟷ Neutral", "#FFC107"  # Amber
    
    def fetch_news(self):
        # Clear previous news
        for widget in self.news_frame.winfo_children():
            widget.destroy()
        
        # Get yesterday's date
        yesterday = datetime.now() - timedelta(days=1)
        date_from = yesterday.strftime("%Y-%m-%d")
        
        # Polygon API endpoint for market news
        url = f"https://api.polygon.io/v2/reference/news?limit=10&order=desc&sort=published_utc&apiKey={self.api_key}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if response.status_code == 200 and 'results' in data:
                # Filter for major news (those with tickers mentioned)
                major_news = [item for item in data['results'] if item.get('tickers') and len(item.get('tickers', [])) > 0]
                
                if not major_news:
                    no_news_label = tk.Label(self.news_frame, text="No major stock news found", 
                                           font=("Helvetica", 14), bg="#f0f0f0")
                    no_news_label.pack(pady=20)
                    return
                
                # Create a canvas with scrollbar
                canvas = tk.Canvas(self.news_frame, bg="#f0f0f0")
                scrollbar = ttk.Scrollbar(self.news_frame, orient="vertical", command=canvas.yview)
                scrollable_frame = ttk.Frame(canvas)
                
                scrollable_frame.bind(
                    "<Configure>",
                    lambda e: canvas.configure(
                        scrollregion=canvas.bbox("all")
                    )
                )
                
                canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
                canvas.configure(yscrollcommand=scrollbar.set)
                
                canvas.pack(side="left", fill="both", expand=True)
                scrollbar.pack(side="right", fill="y")
                
                # Add news items to the scrollable frame
                for i, news in enumerate(major_news[:10]):  # Limit to 10 items
                    # Create a frame for each news item
                    news_item_frame = ttk.Frame(scrollable_frame)
                    news_item_frame.pack(fill="x", padx=5, pady=5)
                    
                    # Get tickers
                    tickers = ", ".join(news.get('tickers', []))
                    
                    # Get expected outcome
                    outcome_text, outcome_color = self.predict_outcome(news.get('title', ''))
                    
                    # Add ticker labels
                    ticker_label = tk.Label(news_item_frame, text=f"Tickers: {tickers}", 
                                         font=("Helvetica", 10, "bold"))
                    ticker_label.pack(anchor="w")
                    
                    # Add headline as a clickable link
                    headline_text = news.get('title', 'No headline available')
                    headline_label = tk.Label(news_item_frame, text=headline_text, 
                                           font=("Helvetica", 12), wraplength=700, 
                                           cursor="hand2")
                    headline_label.pack(anchor="w", pady=2)
                    headline_label.bind("<Button-1>", lambda e, url=news.get('article_url'): self.open_article(url))
                    
                    # Add expected outcome
                    outcome_label = tk.Label(news_item_frame, text=outcome_text, 
                                          font=("Helvetica", 10, "bold"), fg=outcome_color)
                    outcome_label.pack(anchor="w")
                    
                    # Add a separator
                    if i < len(major_news) - 1:
                        separator = ttk.Separator(scrollable_frame, orient='horizontal')
                        separator.pack(fill='x', padx=5, pady=5)
            
            else:
                error_label = tk.Label(self.news_frame, text=f"Error: {data.get('error', 'Unknown error')}", 
                                     font=("Helvetica", 14), bg="#f0f0f0", fg="red")
                error_label.pack(pady=20)
        
        except Exception as e:
            error_label = tk.Label(self.news_frame, text=f"Error: {str(e)}", 
                                 font=("Helvetica", 14), bg="#f0f0f0", fg="red")
            error_label.pack(pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    app = NewsApp(root, api_key="9skphQ6G7_rESW6iTNJDIAycT9gncpje")
    root.mainloop()
