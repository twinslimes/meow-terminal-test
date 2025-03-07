import requests
import re
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
import webbrowser
import calendar
from datetime import date

class NewsApp:
    def __init__(self, root, api_key):
        self.root = root
        self.api_key = api_key
        self.root.title("Stock News Dashboard")
        self.root.geometry("800x600")
        
        # Configure the main window
        self.root.configure(bg="#f0f0f0")
        
        # Create a main frame with scrollbar
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a canvas for scrolling
        self.canvas = tk.Canvas(self.main_frame, bg="#f0f0f0")
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        
        # Configure canvas
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create content frame inside canvas
        self.content_frame = tk.Frame(self.canvas, bg="#f0f0f0")
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        
        # Configure canvas to resize with window
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        self.content_frame.bind('<Configure>', self.on_frame_configure)
        
        # Add a title to content frame
        title_label = tk.Label(self.content_frame, text="Major Stock News", font=("Helvetica", 24), bg="#f0f0f0")
        title_label.pack(pady=20)
        
        # Add market calendar
        self.add_market_calendar()
        
        # Add a refresh button 
        refresh_button = tk.Button(self.content_frame, text="Refresh News", command=self.fetch_news, 
                                  font=("Helvetica", 12), bg="#4CAF50", fg="white")
        refresh_button.pack(pady=10)
        
        # Create a frame for the news items
        self.news_frame = tk.Frame(self.content_frame, bg="#f0f0f0")
        self.news_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Initial news fetch
        self.fetch_news()
    
    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_frame, width=event.width)
    
    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def add_market_calendar(self):
        """Add a weekly stock market calendar to the dashboard"""
        # Create a calendar frame
        calendar_frame = tk.Frame(self.content_frame, bg="#f0f0f0", bd=2, relief=tk.GROOVE)
        calendar_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Add calendar title
        current_date = datetime.now()
        cal_title = tk.Label(calendar_frame, 
                            text=f"Market Schedule - Week of {current_date.strftime('%B %d, %Y')}", 
                            font=("Helvetica", 14, "bold"), 
                            bg="#f0f0f0")
        cal_title.pack(pady=5)
        
        # Create a frame for the weekday grid
        weekdays_frame = tk.Frame(calendar_frame, bg="#f0f0f0")
        weekdays_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Get the current week's Monday date
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        
        # Create a column for each weekday
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        day_frames = []
        
        for i, day in enumerate(weekdays):
            day_date = monday + timedelta(days=i)
            
            # Create frame for this day
            day_frame = tk.Frame(weekdays_frame, bg="#f0f0f0", bd=1, relief=tk.GROOVE, width=150, height=100)
            day_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            day_frame.pack_propagate(False)  # Force the frame to keep its size
            
            # Add day header
            day_header = tk.Label(day_frame, 
                                text=f"{day}\n{day_date.strftime('%m/%d')}", 
                                font=("Helvetica", 10, "bold"), 
                                bg="#f0f0f0")
            day_header.pack(fill=tk.X, pady=2)
            
            # Separator
            ttk.Separator(day_frame, orient='horizontal').pack(fill='x')
            
            # Add this frame to our list
            day_frames.append(day_frame)
        
        # Add sample market events to the calendar
        # In a real app, these would come from an API or database
        events = self.get_market_events(monday)
        
        # Add events to the appropriate day frames
        for event in events:
            event_day = event['day']  # 0 = Monday, 1 = Tuesday, etc.
            if event_day < 5:  # Only weekdays
                event_text = tk.Label(day_frames[event_day], 
                                    text=event['time'] + ' ' + event['name'],
                                    font=("Helvetica", 8),
                                    fg=event['color'],
                                    bg="#f0f0f0",
                                    anchor="w",
                                    justify=tk.LEFT,
                                    wraplength=145)
                event_text.pack(fill=tk.X, padx=2, pady=1, anchor="w")
    
    def get_market_events(self, monday_date):
        """Generate market events for the current week
        This would typically come from an API, but we'll use sample data"""
        # Format: day (0=Monday, 4=Friday), name, time, color
        # Real implementation would fetch from API
        events = [
            {'day': 0, 'name': 'Market Open', 'time': '9:30 AM', 'color': 'black'},
            {'day': 0, 'name': 'XYZ Earnings', 'time': 'After Close', 'color': 'green'},
            {'day': 1, 'name': 'CPI Data Release', 'time': '8:30 AM', 'color': 'orange'},
            {'day': 1, 'name': 'ABC Earnings Call', 'time': '5:00 PM', 'color': 'green'},
            {'day': 2, 'name': 'FOMC Minutes', 'time': '2:00 PM', 'color': 'red'},
            {'day': 2, 'name': 'DEF/GHI Earnings', 'time': 'Pre-Market', 'color': 'green'},
            {'day': 3, 'name': 'Jobless Claims', 'time': '8:30 AM', 'color': 'orange'},
            {'day': 3, 'name': 'JKL Earnings', 'time': 'After Close', 'color': 'green'},
            {'day': 4, 'name': 'PMI Data', 'time': '9:45 AM', 'color': 'orange'},
            {'day': 4, 'name': 'MNO Earnings', 'time': 'Pre-Market', 'color': 'green'}
        ]
        
        # Check if current ticker has earnings this week
        # (In a real implementation, this would check against actual earnings calendar)
        return events
    
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
