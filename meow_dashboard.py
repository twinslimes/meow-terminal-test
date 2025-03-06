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
        
        # Add a title
        title_label = tk.Label(root, text="Major Stock News", font=("Helvetica", 24), bg="#f0f0f0")
        title_label.pack(pady=20)
        
        # Add a refresh button
        refresh_button = tk.Button(root, text="Refresh News", command=self.fetch_news, 
                                  font=("Helvetica", 12), bg="#4CAF50", fg="white")
        refresh_button.pack(pady=10)
        
        # Create a frame for the news items
        self.news_frame = tk.Frame(root, bg="#f0f0f0")
        self.news_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
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