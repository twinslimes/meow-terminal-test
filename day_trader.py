import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import time
from datetime import datetime, timedelta
import pytz
import warnings
import threading

# Import only what we need
from data_utils import calculate_technical_indicators
from models import StockData, GeometricBrownianMotion

# Suppress warnings
warnings.filterwarnings('ignore')

# Version marker for troubleshooting
__version__ = "3.0.0"

def init_day_trader_state():
    """Initialize session state for the day trader module."""
    if 'chart_data' not in st.session_state:
        st.session_state.chart_data = None
    if 'current_price' not in st.session_state:
        st.session_state.current_price = None
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'prediction_accuracy' not in st.session_state:
        st.session_state.prediction_accuracy = []
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 30  # seconds
    if 'timeframe' not in st.session_state:
        st.session_state.timeframe = '5m'
    if 'refresh_thread' not in st.session_state:
        st.session_state.refresh_thread = None
    if 'stop_refresh' not in st.session_state:
        st.session_state.stop_refresh = False

def fetch_stock_data(ticker, timeframe='5m'):
    """Fetch live stock data with simple error handling."""
    try:
        # Map display names to yfinance intervals
        intervals = {
            '30s': '30s',
            '1m': '1m', 
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h'
        }
        
        # Set appropriate period based on timeframe
        if timeframe in ['30s', '1m']:
            period = "2d"  # Shorter period for high-frequency data
        elif timeframe in ['5m', '15m']:
            period = "5d"
        else:
            period = "10d"
            
        # Fetch data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=intervals[timeframe])
        
        # Handle empty data
        if data.empty:
            # Try a fallback to 5m if we got empty data
            if timeframe in ['30s', '1m']:
                st.warning(f"No data available for {timeframe} timeframe. Trying 5m instead.")
                timeframe = '5m'
                st.session_state.timeframe = '5m'
                data = stock.history(period="5d", interval='5m')
            
            if data.empty:
                st.error(f"No data available for {ticker}. Market may be closed.")
                return None
        
        # Store current price and last update time
        if not data.empty:
            st.session_state.current_price = data['Close'].iloc[-1]
            st.session_state.last_update = datetime.now()
            
            # Convert index to Eastern time
            if data.index.tzinfo is not None:
                data.index = data.index.tz_convert(pytz.timezone('US/Eastern'))
                
        return data
    
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def create_live_chart(data, ticker, predictions=None):
    """Create a clean, user-friendly live trading chart."""
    if data is None or data.empty:
        return None
    
    # Create figure with subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                      vertical_spacing=0.05, 
                      row_heights=[0.7, 0.3],
                      subplot_titles=("Price Chart", "Volume & Indicators"))
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in data.iterrows()]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Add simple moving averages
    data['SMA9'] = data['Close'].rolling(window=9).mean()
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['SMA9'],
            line=dict(color='purple', width=1),
            name="9-period MA"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['SMA20'],
            line=dict(color='blue', width=1),
            name="20-period MA"
        ),
        row=1, col=1
    )
    
    # Add VWAP (Volume Weighted Average Price)
    # Reset cumulative sums if new trading day
    if len(data) > 0:
        # Group by trading day
        data['date'] = data.index.date
        vwap_data = []
        
        for date, group in data.groupby('date'):
            # Calculate VWAP for this day
            cumulative_pv = (group['Close'] * group['Volume']).cumsum()
            cumulative_volume = group['Volume'].cumsum()
            
            # Avoid division by zero
            group_vwap = cumulative_pv / cumulative_volume.replace(0, np.nan).fillna(1)
            vwap_data.append(group_vwap)
        
        data['VWAP'] = pd.concat(vwap_data)
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['VWAP'],
                line=dict(color='orange', width=1, dash='dot'),
                name="VWAP"
            ),
            row=1, col=1
        )
    
    # Add predictions if available
    if predictions is not None and len(predictions) > 0:
        # Make sure we have data to work with
        if len(data) > 0:
            last_time = data.index[-1]
            
            # Calculate interval in minutes based on timeframe
            timeframe_minutes = {
                '30s': 0.5,
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '30m': 30,
                '1h': 60
            }.get(st.session_state.timeframe, 5)
            
            # Generate future timestamps for predictions
            future_times = [last_time + timedelta(minutes=timeframe_minutes * (i+1)) for i in range(len(predictions))]
            
            # Add prediction line
            fig.add_trace(
                go.Scatter(
                    x=[last_time] + future_times,
                    y=[data['Close'].iloc[-1]] + predictions,
                    line=dict(color='#4a90e2', width=2, dash='dot'),
                    name="Prediction"
                ),
                row=1, col=1
            )
    
    # Update layout for terminal theme
    fig.update_layout(
        title=f"{ticker} Live Chart - Last Updated: {st.session_state.last_update.strftime('%H:%M:%S') if st.session_state.last_update else 'No data'}",
        paper_bgcolor='#2f2f2f',
        plot_bgcolor='#2f2f2f',
        font=dict(color='#ffffff'),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600
    )
    
    # Update axes appearance
    fig.update_xaxes(gridcolor='#4a90e2', zerolinecolor='#e6f3ff')
    fig.update_yaxes(gridcolor='#4a90e2', zerolinecolor='#e6f3ff')
    
    return fig

def generate_simple_prediction(ticker, steps=12):
    """Generate a simple price prediction using GBM."""
    try:
        if 'chart_data' not in st.session_state or st.session_state.chart_data is None:
            return None
        
        # Get current price
        current_price = st.session_state.current_price
        if not current_price:
            return None
        
        # Create a StockData object
        stock_data = StockData(ticker, "", "")
        stock_data.price = current_price
        
        # Calculate volatility from recent data
        data = st.session_state.chart_data
        returns = data['Close'].pct_change().dropna()
        if len(returns) > 1:
            # Annualize volatility based on timeframe
            tf_minutes = {'30s': 0.5, '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60}.get(st.session_state.timeframe, 5)
            periods_per_year = 252 * 6.5 * 60 / tf_minutes  # Trading days * hours * minutes / timeframe
            stock_data.volatility = returns.std() * np.sqrt(periods_per_year)
        else:
            stock_data.volatility = 0.3  # Default
        
        stock_data.risk_free_rate = 0.05  # Default value
        
        # Use GBM for simple prediction
        model = GeometricBrownianMotion(stock_data)
        
        # Convert timeframe to fraction of a year for T
        timeframe_in_years = {'30s': 0.5/(252*6.5*60), '1m': 1/(252*6.5*60), 
                             '5m': 5/(252*6.5*60), '15m': 15/(252*6.5*60),
                             '30m': 30/(252*6.5*60), '1h': 1/(252*6.5)}.get(st.session_state.timeframe, 5/(252*6.5*60))
        
        T = steps * timeframe_in_years
        dt = timeframe_in_years
        
        # Run simulation with 500 paths
        result = model.simulate(T, dt, 500, current_price * 1.01)
        
        # Extract predictions for each step
        paths = result.get('paths', None)
        predictions = []
        
        if paths is not None and paths.shape[1] > 1:
            for i in range(1, min(steps + 1, paths.shape[1])):
                predictions.append(np.mean(paths[:, i]))
        else:
            # Fallback: linear interpolation to final mean price
            mean_price = result['mean_price']
            for i in range(1, steps + 1):
                predictions.append(current_price + (mean_price - current_price) * (i/steps))
        
        # Store prediction with timestamp for later evaluation
        st.session_state.predictions = predictions
        
        # Add to prediction history
        st.session_state.prediction_history.append({
            'time': datetime.now(),
            'price': current_price,
            'predictions': predictions,
            'timeframe': st.session_state.timeframe
        })
        
        # Keep only the latest 50 predictions
        if len(st.session_state.prediction_history) > 50:
            st.session_state.prediction_history = st.session_state.prediction_history[-50:]
            
        return predictions
        
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        return None

def update_prediction_accuracy():
    """Check past predictions against actual prices to track accuracy."""
    if not st.session_state.prediction_history:
        return
    
    # Get the latest price data
    data = st.session_state.chart_data
    if data is None or data.empty:
        return
    
    # Check each prediction that hasn't been evaluated yet
    for pred in st.session_state.prediction_history:
        if 'evaluated' in pred and pred['evaluated']:
            continue
            
        # Skip if prediction is too recent (not enough time has passed)
        timeframe_minutes = {'30s': 0.5, '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60}.get(pred['timeframe'], 5)
        if datetime.now() - pred['time'] < timedelta(minutes=timeframe_minutes * len(pred['predictions'])):
            continue
            
        # Find actual prices after prediction time
        pred_time = pred['time']
        
        # Convert prediction time to match data timezone if needed
        if data.index.tz and pred_time.tzinfo is None:
            pred_time = pytz.timezone('US/Eastern').localize(pred_time)
            
        # Get actual prices
        after_indices = data.index[data.index > pred_time]
        if len(after_indices) < 1:
            continue
        
        # Get actual close prices
        actual_prices = []
        for i in range(min(len(pred['predictions']), len(after_indices))):
            if i < len(after_indices):
                actual_prices.append(data.loc[after_indices[i], 'Close'])
        
        # Check direction accuracy (up or down)
        if actual_prices and len(pred['predictions']) > 0:
            pred_direction = 1 if pred['predictions'][-1] > pred['price'] else -1
            actual_direction = 1 if actual_prices[-1] > pred['price'] else -1
            
            # Record if prediction was correct
            direction_correct = (pred_direction == actual_direction)
            
            # Update accuracy tracking
            st.session_state.prediction_accuracy.append(direction_correct)
            
            # Keep only latest 20 accuracy records
            if len(st.session_state.prediction_accuracy) > 20:
                st.session_state.prediction_accuracy = st.session_state.prediction_accuracy[-20:]
                
            # Mark prediction as evaluated
            pred['evaluated'] = True
            pred['direction_correct'] = direction_correct

def auto_refresh_thread(ticker, interval):
    """Background thread to refresh data automatically."""
    stop_event = threading.Event()
    st.session_state.stop_refresh = stop_event
    
    while not stop_event.is_set():
        try:
            # Fetch new data
            data = fetch_stock_data(ticker, st.session_state.timeframe)
            
            if data is not None and not data.empty:
                # Update chart data
                st.session_state.chart_data = data
                
                # Generate new prediction
                generate_simple_prediction(ticker)
                
                # Update accuracy metrics
                update_prediction_accuracy()
            
            # Wait for next refresh
            time.sleep(interval)
        except Exception as e:
            print(f"Error in auto-refresh: {e}")
            time.sleep(5)  # Wait a bit on error
    
    print("Auto-refresh stopped")

def display_day_trader_section(ticker):
    """Display a simplified, user-friendly day trader dashboard."""
    st.header(f"📊 Live Trading Dashboard")
    
    # Initialize session state
    init_day_trader_state()
    
    # Main layout - chart on left, controls on right
    col1, col2 = st.columns([4, 1])
    
    # Trading controls sidebar
    with col2:
        st.subheader("⚙️ Trading Controls")
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Chart Timeframe",
            options=['5m', '1m', '15m', '30m', '1h', '30s'],
            index=['5m', '1m', '15m', '30m', '1h', '30s'].index(st.session_state.timeframe)
        )
        
        # Update timeframe if changed
        if timeframe != st.session_state.timeframe:
            st.session_state.timeframe = timeframe
            st.session_state.chart_data = None  # Force refresh
        
        # Fetch data button
        if st.button("🔄 Refresh Data", use_container_width=True):
            with st.spinner("Fetching live data..."):
                data = fetch_stock_data(ticker, timeframe)
                if data is not None:
                    st.session_state.chart_data = data
                    
                    # Generate prediction
                    with st.spinner("Generating prediction..."):
                        generate_simple_prediction(ticker)
        
        # Auto-refresh toggle
        auto_refresh = st.toggle("🔄 Auto-Refresh", value=st.session_state.auto_refresh)
        
        if auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = auto_refresh
            
            if auto_refresh:
                # Stop existing thread if any
                if 'stop_refresh' in st.session_state and st.session_state.stop_refresh:
                    st.session_state.stop_refresh.set()
                
                # Start new thread
                refresh_interval = st.session_state.refresh_interval
                thread = threading.Thread(
                    target=auto_refresh_thread,
                    args=(ticker, refresh_interval),
                    daemon=True
                )
                thread.start()
                st.success(f"Auto-refresh enabled - updating every {refresh_interval} seconds")
            else:
                if 'stop_refresh' in st.session_state and st.session_state.stop_refresh:
                    st.session_state.stop_refresh.set()
                    st.warning("Auto-refresh disabled")
        
        # Only show interval slider when auto-refresh is on
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh Interval (sec)",
                min_value=10,
                max_value=120,
                value=st.session_state.refresh_interval,
                step=10
            )
            
            if refresh_interval != st.session_state.refresh_interval:
                st.session_state.refresh_interval = refresh_interval
                # Restart thread with new interval
                if 'stop_refresh' in st.session_state and st.session_state.stop_refresh:
                    st.session_state.stop_refresh.set()
                    time.sleep(1)  # Wait for old thread to stop
                
                # Start new thread
                thread = threading.Thread(
                    target=auto_refresh_thread,
                    args=(ticker, refresh_interval),
                    daemon=True
                )
                thread.start()
                st.success(f"Refresh interval updated to {refresh_interval} seconds")
        
        # Show current price if available
        if st.session_state.current_price is not None:
            st.metric(
                label="Current Price",
                value=f"${st.session_state.current_price:.2f}"
            )
        
        # Show prediction accuracy if available
        if st.session_state.prediction_accuracy:
            correct = sum(1 for x in st.session_state.prediction_accuracy if x)
            total = len(st.session_state.prediction_accuracy)
            accuracy = (correct / total) * 100 if total > 0 else 0
            
            st.metric(
                label="Prediction Accuracy",
                value=f"{accuracy:.1f}%",
                delta=f"{correct}/{total} correct"
            )
        
        # Generate prediction button
        if st.button("🔮 Generate Prediction", use_container_width=True):
            if st.session_state.chart_data is not None:
                with st.spinner("Generating prediction..."):
                    generate_simple_prediction(ticker)
                st.success("New prediction generated!")
            else:
                st.error("Please fetch data first")
    
    # Main chart area
    with col1:
        if st.session_state.chart_data is not None:
            # Create live chart
            fig = create_live_chart(
                st.session_state.chart_data,
                ticker,
                st.session_state.predictions if 'predictions' in st.session_state else None
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Show key indicators if we have data
            if len(st.session_state.chart_data) >= 14:
                # Calculate indicators
                indicators = calculate_technical_indicators(st.session_state.chart_data)
                
                # Only show if we have indicators calculated
                if not indicators.empty:
                    latest = indicators.iloc[-1]
                    
                    # Create indicator cards
                    st.subheader("📈 Key Trading Indicators")
                    cols = st.columns(3)
                    
                    # RSI
                    with cols[0]:
                        if 'RSI' in latest:
                            rsi_value = latest['RSI']
                            rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                            rsi_color = "red" if rsi_value > 70 else "green" if rsi_value < 30 else "gray"
                            
                            st.metric(
                                label="RSI (14)",
                                value=f"{rsi_value:.2f}",
                                delta=rsi_status,
                                delta_color="off"
                            )
                            st.markdown(f"<span style='color:{rsi_color};'>{rsi_status}</span>", unsafe_allow_html=True)
                    
                    # MACD
                    with cols[1]:
                        if 'MACD' in latest and 'MACD_Signal' in latest:
                            macd_value = latest['MACD']
                            signal_value = latest['MACD_Signal']
                            macd_diff = macd_value - signal_value
                            macd_status = "Bullish" if macd_diff > 0 else "Bearish"
                            macd_color = "green" if macd_diff > 0 else "red"
                            
                            st.metric(
                                label="MACD",
                                value=f"{macd_value:.4f}",
                                delta=f"{macd_diff:.4f}",
                                delta_color="normal" if macd_diff > 0 else "inverse"
                            )
                            st.markdown(f"<span style='color:{macd_color};'>{macd_status}</span>", unsafe_allow_html=True)
                    
                    # Bollinger Bands
                    with cols[2]:
                        if all(band in latest for band in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                            current_price = latest['Close']
                            bb_upper = latest['BB_Upper']
                            bb_lower = latest['BB_Lower']
                            
                            # %B indicator
                            pct_b = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5
                            bb_status = "Upper Band" if pct_b > 0.8 else "Lower Band" if pct_b < 0.2 else "Middle"
                            bb_color = "red" if pct_b > 0.8 else "green" if pct_b < 0.2 else "gray"
                            
                            st.metric(
                                label="Bollinger %B",
                                value=f"{pct_b:.2f}",
                                delta=bb_status,
                                delta_color="off"
                            )
                            st.markdown(f"<span style='color:{bb_color};'>{bb_status}</span>", unsafe_allow_html=True)
        else:
            # Show instructions to fetch data
            st.info(f"👆 Click 'Refresh Data' to load the live chart for {ticker}")
            
            # Show example/placeholder
            st.markdown("""
            ### Live Trading Features:
            
            - **Real-time price data** with customizable timeframes
            - **Auto-refreshing chart** to keep you up to date
            - **Price predictions** that improve over time
            - **Key technical indicators** for day trading decisions
            
            Click the refresh button to get started!
            """)
