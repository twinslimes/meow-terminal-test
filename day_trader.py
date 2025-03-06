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
import queue
from scipy import stats
import requests

# Import local modules
from data_utils import calculate_technical_indicators, generate_technical_signals
from models import (StockData, GeometricBrownianMotion, JumpDiffusionModel, 
                   HestonModel, VarianceGammaModel, QuasiMonteCarloModel)

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize session state keys for persistent storage
def init_session_state():
    """Initialize session state variables if they don't exist."""
    if 'live_chart_data' not in st.session_state:
        st.session_state.live_chart_data = None
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'prediction_accuracy' not in st.session_state:
        st.session_state.prediction_accuracy = []
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'model_weights' not in st.session_state:
        # Initialize with equal weights
        st.session_state.model_weights = {
            'GeometricBrownianMotion': 0.2,
            'JumpDiffusionModel': 0.2, 
            'HestonModel': 0.2,
            'VarianceGammaModel': 0.2,
            'QuasiMonteCarloModel': 0.2
        }
    if 'model_performance' not in st.session_state:
        # Track each model's performance for dynamic weight adjustment
        st.session_state.model_performance = {
            'GeometricBrownianMotion': {'correct': 0, 'total': 0},
            'JumpDiffusionModel': {'correct': 0, 'total': 0},
            'HestonModel': {'correct': 0, 'total': 0},
            'VarianceGammaModel': {'correct': 0, 'total': 0},
            'QuasiMonteCarloModel': {'correct': 0, 'total': 0}
        }
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 60  # seconds
    if 'selected_timeframe' not in st.session_state:
        st.session_state.selected_timeframe = '1m'  # Default timeframe
    if 'day_trader_ticker' not in st.session_state:
        st.session_state.day_trader_ticker = None
    if 'refresh_thread' not in st.session_state:
        st.session_state.refresh_thread = None
    if 'stop_refresh' not in st.session_state:
        st.session_state.stop_refresh = False
    if 'risk_free_rate' not in st.session_state:
        st.session_state.risk_free_rate = 0.04  # Default value, will try to fetch current rate
    if 'data_range_days' not in st.session_state:
        # Default data range based on timeframe - will be adjusted automatically
        st.session_state.data_range_days = 5
    if 'custom_horizon' not in st.session_state:
        # Default prediction horizon based on timeframe - will be adjusted automatically
        st.session_state.custom_horizon = 15

def get_risk_free_rate():
    """Fetch current risk-free rate from FRED API."""
    try:
        # Try to get from FRED API - Treasury 3-Month yield is a good proxy for risk-free rate
        url = "https://api.stlouisfed.org/fred/series/observations?series_id=DTB3&api_key=407359595d242cb6848578f701b78f83&file_type=json&limit=1&sort_order=desc"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if 'observations' in data and len(data['observations']) > 0:
                rate_str = data['observations'][0]['value']
                if rate_str != '.':  # Check for missing value
                    rate = float(rate_str) / 100  # Convert percentage to decimal
                    return rate
        
        # If API call fails, use default rate
        return st.session_state.risk_free_rate
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        return st.session_state.risk_free_rate

def get_appropriate_data_range(timeframe):
    """Calculate appropriate data range based on timeframe."""
    # Map timeframes to appropriate data range in days
    # More granular timeframes need shorter periods to avoid huge datasets
    timeframe_ranges = {
        '30s': 1,   # 1 day for 30-second data
        '1m': 3,    # 3 days for 1-minute data
        '5m': 7,    # 7 days for 5-minute data
        '15m': 14,  # 14 days for 15-minute data
        '30m': 30,  # 30 days for 30-minute data
        '1h': 60    # 60 days for hourly data
    }
    
    return timeframe_ranges.get(timeframe, 5)  # Default to 5 days if timeframe not recognized

def get_default_prediction_horizon(timeframe):
    """Get default prediction horizon based on timeframe."""
    # Map timeframes to appropriate prediction horizons
    # Shorter timeframes need fewer steps
    timeframe_horizons = {
        '30s': 5,    # 5 steps for 30-second data (2.5 minutes)
        '1m': 10,    # 10 steps for 1-minute data (10 minutes)
        '5m': 12,    # 12 steps for 5-minute data (1 hour)
        '15m': 16,   # 16 steps for 15-minute data (4 hours)
        '30m': 16,   # 16 steps for 30-minute data (8 hours)
        '1h': 24     # 24 steps for hourly data (1 day)
    }
    
    return timeframe_horizons.get(timeframe, 15)  # Default to 15 steps if timeframe not recognized

def fetch_live_data(ticker, timeframe='1m', custom_days=None):
    """Fetch live intraday data for the selected ticker."""
    try:
        with st.spinner(f"Fetching live data for {ticker}..."):
            # Map user-friendly timeframe names to yfinance intervals
            timeframe_map = {
                '30s': '30s',
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h'
            }
            
            interval = timeframe_map.get(timeframe, '1m')
            
            # Determine appropriate data range based on timeframe
            if custom_days is not None:
                days = custom_days
            else:
                days = get_appropriate_data_range(timeframe)
            
            # Update the stored data range
            st.session_state.data_range_days = days
            
            # Use yfinance to get intraday data
            stock = yf.Ticker(ticker)
            
            # For very small timeframes like 30s, we need to use the 'period' parameter
            if timeframe == '30s':
                intraday_data = stock.history(period=f"{min(days, 7)}d", interval=interval)
            else:
                intraday_data = stock.history(period=f"{days}d", interval=interval)
            
            if len(intraday_data) > 0:
                # If market is closed, we might get data from the last trading day
                # Convert index to US Eastern time for market hours
                eastern = pytz.timezone('US/Eastern')
                intraday_data.index = intraday_data.index.tz_convert(eastern)
                
                # Also fetch daily data for model calibration
                daily_data = stock.history(period="3mo")
                
                # Store both in session state for model calibration
                st.session_state.intraday_data = intraday_data
                st.session_state.daily_data = daily_data
                
                # Get current market price
                current_price = intraday_data['Close'].iloc[-1]
                st.session_state.current_price = current_price
                
                # Store last update time
                st.session_state.last_update_time = datetime.now()
                
                # Store data for live chart
                st.session_state.live_chart_data = intraday_data
                
                # Update ticker in session state
                st.session_state.day_trader_ticker = ticker
                
                # Update default prediction horizon based on new timeframe
                if st.session_state.custom_horizon == get_default_prediction_horizon(st.session_state.selected_timeframe):
                    # Only update if user hasn't manually changed it
                    st.session_state.custom_horizon = get_default_prediction_horizon(timeframe)
                
                # Fetch current risk-free rate
                current_rate = get_risk_free_rate()
                if current_rate != st.session_state.risk_free_rate:
                    st.session_state.risk_free_rate = current_rate
                
                return intraday_data
            else:
                st.error("No intraday data available. Market may be closed.")
                return None
    except Exception as e:
        st.error(f"Error fetching live data: {e}")
        return None

def create_live_chart(data, ticker, predictions=None):
    """Create an interactive live chart with candlesticks and indicators."""
    # Create subplots with price, volume, and indicators
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                      vertical_spacing=0.05, row_heights=[0.5, 0.2, 0.3],
                      subplot_titles=("Price & Predictions", "Volume", "Indicators"))
    
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
    
    # Add volume bar chart
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
    
    # Calculate and add moving averages
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    # Add MAs to chart
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['EMA_9'],
            line=dict(color='purple', width=1),
            name="EMA 9"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            line=dict(color='blue', width=1),
            name="SMA 20"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['VWAP'],
            line=dict(color='orange', width=1, dash='dot'),
            name="VWAP"
        ),
        row=1, col=1
    )
    
    # Calculate technical indicators
    if len(data) >= 14:  # Need at least 14 periods for RSI
        # RSI
        delta = data['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Add RSI chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                line=dict(color='red', width=1),
                name="RSI (14)"
            ),
            row=3, col=1
        )
        
        # Add RSI overbought/oversold lines
        fig.add_trace(
            go.Scatter(
                x=[data.index[0], data.index[-1]],
                y=[70, 70],
                line=dict(color='red', width=1, dash='dash'),
                name="Overbought"
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[data.index[0], data.index[-1]],
                y=[30, 30],
                line=dict(color='green', width=1, dash='dash'),
                name="Oversold"
            ),
            row=3, col=1
        )
    
    # Add MACD if enough data
    if len(data) >= 26:
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # Add MACD line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                line=dict(color='blue', width=1),
                name="MACD",
                visible='legendonly'  # Hide by default in legend
            ),
            row=3, col=1
        )
        
        # Add MACD signal line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                line=dict(color='orange', width=1),
                name="Signal",
                visible='legendonly'  # Hide by default in legend
            ),
            row=3, col=1
        )
    
    # Add predictions if available
    if predictions is not None and len(predictions) > 0:
        # Extract the last actual timestamp
        last_time = data.index[-1]
        
        # Calculate interval in minutes based on timeframe
        timeframe_minutes = {
            '30s': 0.5,  # 0.5 minutes (30 seconds)
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60
        }.get(st.session_state.selected_timeframe, 1)
        
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
        
        # Add prediction confidence interval if available
        if 'prediction_upper' in st.session_state and 'prediction_lower' in st.session_state:
            fig.add_trace(
                go.Scatter(
                    x=future_times,
                    y=st.session_state.prediction_upper,
                    line=dict(width=0),
                    marker=dict(color="#444"),
                    showlegend=False,
                    name="Upper Bound"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=future_times,
                    y=st.session_state.prediction_lower,
                    line=dict(width=0),
                    marker=dict(color="#444"),
                    fillcolor='rgba(74, 144, 226, 0.3)',
                    fill='tonexty',
                    showlegend=False,
                    name="Lower Bound"
                ),
                row=1, col=1
            )
    
    # Update layout for terminal theme
    fig.update_layout(
        title=f"{ticker} Live Trading Chart - {st.session_state.last_update_time.strftime('%H:%M:%S') if st.session_state.last_update_time else 'No data'}",
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
        height=800
    )
    
    # Update axes appearance
    fig.update_xaxes(
        gridcolor='#4a90e2',
        zerolinecolor='#e6f3ff',
        row=1, col=1
    )
    
    fig.update_yaxes(
        gridcolor='#4a90e2',
        zerolinecolor='#e6f3ff',
        row=1, col=1
    )
    
    fig.update_xaxes(
        gridcolor='#4a90e2',
        zerolinecolor='#e6f3ff',
        row=2, col=1
    )
    
    fig.update_yaxes(
        gridcolor='#4a90e2',
        zerolinecolor='#e6f3ff',
        row=2, col=1
    )
    
    fig.update_xaxes(
        gridcolor='#4a90e2',
        zerolinecolor='#e6f3ff',
        row=3, col=1
    )
    
    fig.update_yaxes(
        gridcolor='#4a90e2',
        zerolinecolor='#e6f3ff',
        title_text="RSI",
        range=[0, 100],
        row=3, col=1
    )
    
    return fig

def update_model_weights():
    """Update model weights based on their prediction accuracy."""
    if 'model_performance' not in st.session_state:
        return
    
    performance = st.session_state.model_performance
    
    # Calculate accuracy for each model
    accuracies = {}
    for model, perf in performance.items():
        if perf['total'] > 0:
            accuracies[model] = perf['correct'] / perf['total']
        else:
            accuracies[model] = 0.2  # Default 20% if no data
    
    # If all models have zero total, return
    if sum(perf['total'] for perf in performance.values()) == 0:
        return
    
    # Apply softmax to get weights that sum to 1
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    exp_accuracies = {model: np.exp(acc * 2) + epsilon for model, acc in accuracies.items()}  # Multiply by 2 to amplify differences
    total_exp = sum(exp_accuracies.values())
    
    # Update weights - use exponential moving average to smooth transitions
    alpha = 0.3  # Smoothing factor
    for model in st.session_state.model_weights:
        if model in exp_accuracies:
            new_weight = exp_accuracies[model] / total_exp
            st.session_state.model_weights[model] = alpha * new_weight + (1 - alpha) * st.session_state.model_weights[model]

def generate_ensemble_prediction(ticker, horizon=None, num_sims=1000):
    """Generate price predictions using multiple models and ensemble them."""
    try:
        # Check if we have data to work with
        if 'intraday_data' not in st.session_state or st.session_state.intraday_data is None:
            st.error("No intraday data available for prediction.")
            return None
        
        # Use the custom horizon if provided, otherwise use the session state value
        if horizon is None:
            horizon = st.session_state.custom_horizon
        
        # Get current price
        current_price = st.session_state.current_price if 'current_price' in st.session_state else 0
        
        if current_price == 0:
            st.error("Invalid current price for prediction.")
            return None
        
        # Create a StockData object for models
        stock_data = StockData(ticker, "", "")
        stock_data.price = current_price
        
        # Calculate volatility from intraday data
        intraday_returns = st.session_state.intraday_data['Close'].pct_change().dropna()
        intraday_volatility = intraday_returns.std() * np.sqrt(252 * 6.5 * 60)  # Annualized from minute data
        stock_data.volatility = intraday_volatility
        
        # Use current risk-free rate from session state
        stock_data.risk_free_rate = st.session_state.risk_free_rate
        
        # Set historical data for calibration
        stock_data.historical_data = st.session_state.daily_data
        stock_data.returns = stock_data.historical_data['Close'].pct_change().dropna()
        
        # Convert timeframe to minutes for T calculation
        interval_minutes = {
            '30s': 0.5,  # 0.5 minutes (30 seconds)
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60
        }.get(st.session_state.selected_timeframe, 1)
        
        # Initialize models
        models = {
            'GeometricBrownianMotion': GeometricBrownianMotion(stock_data),
            'JumpDiffusionModel': JumpDiffusionModel(stock_data),
            'HestonModel': HestonModel(stock_data),
            'VarianceGammaModel': VarianceGammaModel(stock_data),
            'QuasiMonteCarloModel': QuasiMonteCarloModel(stock_data)
        }
        
        # Calibrate all models
        for model in models.values():
            model.calibrate()
        
        # Run simulations for all models
        model_predictions = {}
        model_upper_bounds = {}
        model_lower_bounds = {}
        
        # Scale T based on the timeframe (convert minutes to years)
        T = (horizon * interval_minutes) / (252 * 6.5 * 60)  # Trading days * hours per day * minutes per hour
        dt = T / horizon
        
        for model_name, model in models.items():
            # Run simulation
            result = model.simulate(T, dt, num_sims, current_price * 1.01)  # Target price doesn't matter here
            
            # Extract prices from the simulation
            if 'paths' in result:
                # Use full paths if available
                paths = result['paths']
                predictions = []
                uppers = []
                lowers = []
                
                for i in range(1, horizon + 1):
                    if i < paths.shape[1]:
                        step_prices = paths[:, i]
                        predictions.append(np.mean(step_prices))
                        uppers.append(np.percentile(step_prices, 95))
                        lowers.append(np.percentile(step_prices, 5))
            else:
                # Fall back to simplified prediction
                final_prices = result['final_prices']
                # Linearly interpolate to create path
                predictions = [current_price + (result['mean_price'] - current_price) * (i/horizon) for i in range(1, horizon + 1)]
                # Create confidence intervals
                uppers = [predictions[i-1] + result['std_price'] * np.sqrt(i/horizon) for i in range(1, horizon + 1)]
                lowers = [predictions[i-1] - result['std_price'] * np.sqrt(i/horizon) for i in range(1, horizon + 1)]
            
            model_predictions[model_name] = predictions
            model_upper_bounds[model_name] = uppers
            model_lower_bounds[model_name] = lowers
        
        # Get current model weights
        weights = st.session_state.model_weights
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            norm_weights = {k: v/weight_sum for k, v in weights.items()}
        else:
            # Default to equal weights if sum is 0
            norm_weights = {k: 1.0/len(weights) for k in weights}
        
        # Compute weighted predictions
        ensemble_predictions = []
        ensemble_uppers = []
        ensemble_lowers = []
        
        for i in range(horizon):
            # Calculate weighted average for each time step
            pred_i = sum(norm_weights[model] * preds[i] 
                         for model, preds in model_predictions.items() 
                         if i < len(preds))
            
            upper_i = sum(norm_weights[model] * uppers[i] 
                          for model, uppers in model_upper_bounds.items() 
                          if i < len(uppers))
            
            lower_i = sum(norm_weights[model] * lowers[i] 
                          for model, lowers in model_lower_bounds.items() 
                          if i < len(lowers))
            
            ensemble_predictions.append(pred_i)
            ensemble_uppers.append(upper_i)
            ensemble_lowers.append(lower_i)
        
        # Store predictions in session state
        st.session_state.predictions = ensemble_predictions
        st.session_state.prediction_upper = ensemble_uppers
        st.session_state.prediction_lower = ensemble_lowers
        st.session_state.prediction_time = datetime.now()
        
        # Store individual model predictions for analysis
        st.session_state.model_predictions = model_predictions
        
        # Store prediction info in history for later accuracy checking
        st.session_state.prediction_history.append({
            'time': st.session_state.prediction_time,
            'price': current_price,
            'predictions': ensemble_predictions,
            'model_predictions': {model: preds for model, preds in model_predictions.items()},
            'horizon': horizon,
            'interval_minutes': interval_minutes,
            'upper': ensemble_uppers,
            'lower': ensemble_lowers
        })
        
        # Keep only the last 100 predictions to manage memory
        if len(st.session_state.prediction_history) > 100:
            st.session_state.prediction_history = st.session_state.prediction_history[-100:]
        
        return ensemble_predictions
    
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def update_prediction_accuracy():
    """Compare past predictions to actual prices and update model weights."""
    if not st.session_state.prediction_history or len(st.session_state.prediction_history) < 2:
        return
    
    # Get the latest price data
    if 'live_chart_data' not in st.session_state or st.session_state.live_chart_data is None:
        return
    
    # Process each prediction in history that hasn't been evaluated yet
    for i, pred_record in enumerate(st.session_state.prediction_history):
        if 'evaluated' in pred_record and pred_record['evaluated']:
            continue
        
        # Skip very recent predictions
        if datetime.now() - pred_record['time'] < timedelta(minutes=pred_record['horizon'] * pred_record['interval_minutes']):
            continue
        
        # Find actual prices after the prediction time
        pred_time = pred_record['time']
        price_data = st.session_state.live_chart_data
        
        # Convert prediction time to match price data index timezone
        if price_data.index.tz:
            if pred_time.tzinfo is None:
                pred_time = pytz.timezone('US/Eastern').localize(pred_time)
        
        # Find index positions after prediction time
        after_indices = price_data.index[price_data.index > pred_time]
        
        if len(after_indices) < pred_record['horizon']:
            continue  # Not enough data yet
        
        # Get actual prices at the prediction horizons
        actual_prices = []
        for j in range(min(pred_record['horizon'], len(after_indices))):
            if j < len(after_indices):
                actual_prices.append(price_data.loc[after_indices[j], 'Close'])
        
        # Compare predictions to actuals
        if len(actual_prices) > 0 and len(pred_record['predictions']) > 0:
            # Calculate errors
            errors = []
            for j in range(min(len(actual_prices), len(pred_record['predictions']))):
                predicted = pred_record['predictions'][j]
                actual = actual_prices[j]
                # Percent error
                pct_error = abs(predicted - actual) / actual * 100
                errors.append(pct_error)
            
            avg_error = sum(errors) / len(errors) if errors else 0
            
            # Direction accuracy (final point)
            if len(actual_prices) > 0 and len(pred_record['predictions']) > 0:
                predicted_direction = 1 if pred_record['predictions'][-1] > pred_record['price'] else -1
                actual_direction = 1 if actual_prices[-1] > pred_record['price'] else -1
                direction_correct = predicted_direction == actual_direction
                
                # Store accuracy metrics
                pred_record['avg_error'] = avg_error
                pred_record['direction_correct'] = direction_correct
                pred_record['evaluated'] = True
                
                # Update overall accuracy tracking
                st.session_state.prediction_accuracy.append(direction_correct)
                
                # Keep only recent accuracy records
                if len(st.session_state.prediction_accuracy) > 50:
                    st.session_state.prediction_accuracy = st.session_state.prediction_accuracy[-50:]
    
                # Now evaluate each model's predictions if available
                if 'model_predictions' in pred_record:
                    for model_name, model_preds in pred_record['model_predictions'].items():
                        if len(model_preds) > 0:
                            # Direction accuracy for this model
                            model_direction = 1 if model_preds[-1] > pred_record['price'] else -1
                            model_correct = model_direction == actual_direction
                            
                            # Update model performance counter
                            if model_name in st.session_state.model_performance:
                                st.session_state.model_performance[model_name]['total'] += 1
                                if model_correct:
                                    st.session_state.model_performance[model_name]['correct'] += 1
    
    # Update model weights based on accuracy
    update_model_weights()

def auto_refresh_data(ticker, interval_seconds=60):
    """Background thread to auto-refresh data."""
    stop_event = threading.Event()
    
    # Store the stop event so we can access it later
    st.session_state.stop_refresh_event = stop_event
    
    while not stop_event.is_set():
        try:
            # Fetch new data
            new_data = fetch_live_data(ticker, st.session_state.selected_timeframe)
            
            # Generate new predictions
            if new_data is not None:
                generate_ensemble_prediction(ticker)
                
                # Update accuracy of past predictions
                update_prediction_accuracy()
                
            # Wait for the next refresh
            stop_event.wait(interval_seconds)
        except Exception as e:
            print(f"Error in auto-refresh thread: {e}")
            # If there's an error, wait a bit before trying again
            stop_event.wait(10)
    
    print("Auto-refresh thread stopping")

def display_day_trader_section(ticker):
    """Display simplified and powerful day trader dashboard."""
    st.header(f"Live Day Trading Analysis")
    
    # Initialize session state variables
    init_session_state()
    
    # Check if ticker has changed
    if st.session_state.day_trader_ticker != ticker and st.session_state.day_trader_ticker is not None:
        st.warning(f"Ticker changed from {st.session_state.day_trader_ticker} to {ticker}. Refreshing data.")
        
        # Stop any running auto-refresh thread
        if 'stop_refresh_event' in st.session_state and st.session_state.stop_refresh_event is not None:
            st.session_state.stop_refresh_event.set()
            time.sleep(1)  # Give thread time to stop
        
        # Clear existing data
        st.session_state.live_chart_data = None
        st.session_state.predictions = []
        st.session_state.prediction_history = []
        st.session_state.day_trader_ticker = ticker
    
    # Main layout with chart and controls
    col1, col2 = st.columns([3, 1])
    
    # Sidebar for intraday trading controls
    with col2:
        st.markdown("<h3 style='color: #e6f3ff;'>Trading Controls</h3>", unsafe_allow_html=True)
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Chart Timeframe",
            options=['30s', '1m', '5m', '15m', '30m', '1h'],
            index=['30s', '1m', '5m', '15m', '30m', '1h'].index(st.session_state.selected_timeframe)
        )
        
        if timeframe != st.session_state.selected_timeframe:
            st.session_state.selected_timeframe = timeframe
            # Clear data to force refresh with new timeframe
            st.session_state.live_chart_data = None
            # Update default prediction horizon
            st.session_state.custom_horizon = get_default_prediction_horizon(timeframe)
        
        # Data range control
        data_range = st.slider(
            "Data Range (days)", 
            min_value=1, 
            max_value=60, 
            value=st.session_state.data_range_days, 
            help="Amount of historical data to fetch"
        )
        
        # Fetch data button
        if st.button("Fetch Live Data", key="fetch_data", use_container_width=True):
            fetch_live_data(ticker, timeframe, custom_days=data_range)
            # Generate initial predictions
            generate_ensemble_prediction(ticker)
        
        # Auto-refresh toggle
        auto_refresh = st.toggle("Auto-Refresh Data", value=st.session_state.auto_refresh)
        
        # Handle auto-refresh state change
        if auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = auto_refresh
            
            if auto_refresh:
                # Start auto-refresh thread
                if 'stop_refresh_event' in st.session_state and st.session_state.stop_refresh_event is not None:
                    st.session_state.stop_refresh_event.set()  # Stop existing thread if any
                
                refresh_thread = threading.Thread(
                    target=auto_refresh_data,
                    args=(ticker, st.session_state.refresh_interval),
                    daemon=True
                )
                refresh_thread.start()
                st.session_state.refresh_thread = refresh_thread
                st.success(f"Auto-refresh enabled - updating every {st.session_state.refresh_interval} seconds")
            else:
                # Stop auto-refresh thread
                if 'stop_refresh_event' in st.session_state and st.session_state.stop_refresh_event is not None:
                    st.session_state.stop_refresh_event.set()
                    st.warning("Auto-refresh disabled")
        
        if auto_refresh:
            # Refresh interval slider only visible when auto-refresh is on
            refresh_interval = st.slider(
                "Refresh Interval (seconds)", 
                min_value=10, 
                max_value=300,
                value=st.session_state.refresh_interval,
                step=10
            )
            
            if refresh_interval != st.session_state.refresh_interval:
                st.session_state.refresh_interval = refresh_interval
                # Restart the thread with new interval
                if 'stop_refresh_event' in st.session_state and st.session_state.stop_refresh_event is not None:
                    st.session_state.stop_refresh_event.set()  # Stop existing thread
                    time.sleep(1)  # Give thread time to stop
                
                # Start new thread with updated interval
                refresh_thread = threading.Thread(
                    target=auto_refresh_data,
                    args=(ticker, refresh_interval),
                    daemon=True
                )
                refresh_thread.start()
                st.session_state.refresh_thread = refresh_thread
                st.success(f"Refresh interval updated to {refresh_interval} seconds")
        
        # Display current price and last update time
        if 'current_price' in st.session_state and 'last_update_time' in st.session_state:
            price_container = st.container()
            with price_container:
                st.metric(
                    label="Current Price",
                    value=f"${st.session_state.current_price:.2f}"
                )
                
                st.markdown(f"""
                <div style='text-align: center; color: #e6f3ff; margin-bottom: 10px;'>
                Last updated: {st.session_state.last_update_time.strftime('%H:%M:%S')}
                </div>
                """, unsafe_allow_html=True)
        
        # Prediction controls
        st.markdown("<h3 style='color: #e6f3ff;'>Prediction Controls</h3>", unsafe_allow_html=True)
        
        # Risk-free rate adjustment (with current value)
        risk_rate = st.number_input(
            "Risk-Free Rate (%)", 
            min_value=0.0, 
            max_value=10.0, 
            value=float(st.session_state.risk_free_rate * 100),
            step=0.1,
            help="Treasury yield used as risk-free rate in models"
        )
        # Update risk free rate in session state (convert from percentage to decimal)
        st.session_state.risk_free_rate = risk_rate / 100
        
        # Prediction horizon
        horizon = st.slider(
            "Prediction Steps", 
            min_value=5, 
            max_value=50, 
            value=st.session_state.custom_horizon, 
            step=1,
            help=f"Number of future {timeframe} periods to predict"
        )
        
        # Update custom horizon in session state
        st.session_state.custom_horizon = horizon
        
        # Display model weights
        with st.expander("Model Weights", expanded=False):
            st.write("Current model weights (updated based on performance):")
            
            for model, weight in st.session_state.model_weights.items():
                # Format model name for display
                display_name = model.replace("Model", "").replace("Geometric", "GBM")
                
                # Calculate accuracy if available
                if model in st.session_state.model_performance:
                    perf = st.session_state.model_performance[model]
                    if perf['total'] > 0:
                        accuracy = perf['correct'] / perf['total'] * 100
                        accuracy_str = f" ({accuracy:.1f}% accuracy)"
                    else:
                        accuracy_str = ""
                else:
                    accuracy_str = ""
                
                # Show progress bar for weight
                st.markdown(f"**{display_name}**: {weight*100:.1f}%{accuracy_str}")
                st.progress(weight)
        
        # Button to generate prediction
        if st.button("Generate Prediction", key="generate_prediction", use_container_width=True):
            with st.spinner("Generating price predictions..."):
                generate_ensemble_prediction(ticker, horizon)
        
        # Show prediction accuracy if available
        if st.session_state.prediction_accuracy and len(st.session_state.prediction_accuracy) > 0:
            correct = sum(1 for x in st.session_state.prediction_accuracy if x)
            total = len(st.session_state.prediction_accuracy)
            accuracy = (correct / total) * 100 if total > 0 else 0
            
            st.markdown(f"""
            <div style='margin-top: 10px; font-size: 16px; color: #e6f3ff; text-align: center;'>
            Prediction Accuracy: {accuracy:.1f}% ({correct}/{total})
            </div>
            """, unsafe_allow_html=True)
            
            # Color based on accuracy
            accuracy_color = "#4CAF50" if accuracy > 70 else "#FFC107" if accuracy > 50 else "#F44336"
            st.markdown(f"""
            <div style='margin: 5px 0; height: 10px; background-color: #444; border-radius: 5px;'>
                <div style='height: 100%; width: {accuracy}%; background-color: {accuracy_color}; border-radius: 5px;'></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Technical signals if data available
        if 'live_chart_data' in st.session_state and st.session_state.live_chart_data is not None:
            st.markdown("<h3 style='color: #e6f3ff;'>Trading Signals</h3>", unsafe_allow_html=True)
            
            # Calculate technical indicators
            data = st.session_state.live_chart_data
            if len(data) >= 14:  # Need enough data for indicators
                indicators = calculate_technical_indicators(data)
                signals = generate_technical_signals(indicators)
                
                # Display summary signal first
                if "Summary" in signals:
                    for signal in signals["Summary"]:
                        signal_color = "green" if signal['direction'] == "bullish" else "red" if signal['direction'] == "bearish" else "gray"
                        st.markdown(f"""
                        <div style='font-size: 18px; color: {signal_color}; text-align: center; margin: 10px 0; 
                             border: 1px solid {signal_color}; padding: 5px; border-radius: 5px;'>
                            {signal['description']}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display category signals in expandable sections
                for category in ["Moving Averages", "Oscillators", "Trend Indicators"]:
                    if category in signals:
                        with st.expander(f"{category}", expanded=category == "Moving Averages"):
                            for signal in signals[category]:
                                signal_color = "green" if signal['direction'] == "bullish" else "red" if signal['direction'] == "bearish" else "gray"
                                st.markdown(f"<span style='color:{signal_color};'>• {signal['description']}</span>", unsafe_allow_html=True)
    
    # Main chart area
    with col1:
        chart_container = st.container()
        with chart_container:
            if 'live_chart_data' in st.session_state and st.session_state.live_chart_data is not None:
                # Create and display live chart
                fig = create_live_chart(
                    st.session_state.live_chart_data, 
                    ticker, 
                    st.session_state.predictions if st.session_state.predictions else None
                )
                st.plotly_chart(fig, use_container_width=True, height=700)
            else:
                # Prompt to fetch data
                st.info(f"Click 'Fetch Live Data' to load intraday chart for {ticker}")
                
                # Example chart image/placeholder
                st.markdown(f"""
                <div style="text-align: center; margin: 20px; padding: 40px; border: 1px dashed #e6f3ff; border-radius: 10px;">
                    <h3 style="color: #e6f3ff;">Live Day Trading Chart</h3>
                    <p style="color: #e6f3ff;">
                        Interactive candlestick chart with multiple timeframes (30s to 1h)<br>
                        Technical indicators and predictive algorithms<br>
                        Auto-refreshing price data for real-time analysis
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Add key metrics below the chart
        if 'live_chart_data' in st.session_state and st.session_state.live_chart_data is not None:
            data = st.session_state.live_chart_data
            
            # Show key trading metrics
            st.markdown("<h3 style='color: #e6f3ff;'>Key Trading Metrics</h3>", unsafe_allow_html=True)
            metrics_cols = st.columns(4)
            
            # Technical indicators for quick reference
            if len(data) >= 14:  # Need enough data for RSI
                indicators = calculate_technical_indicators(data)
                latest = indicators.iloc[-1]
                
                with metrics_cols[0]:
                    # RSI
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
                
                with metrics_cols[1]:
                    # MACD if available
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
                    else:
                        st.metric(
                            label="MACD",
                            value="N/A"
                        )
                
                with metrics_cols[2]:
                    # Bollinger Bands if available
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
                        st.metric(
                            label="Bollinger %B",
                            value="N/A"
                        )
                
                with metrics_cols[3]:
                    # Moving Average Relationship
                    current_price = latest['Close']
                    sma_20 = latest.get('SMA_20', None)
                    sma_50 = latest.get('SMA_50', None)
                    
                    if sma_20 is not None and sma_50 is not None:
                        if current_price > sma_20 > sma_50:
                            ma_status = "Strong Uptrend"
                            ma_color = "green"
                        elif current_price > sma_20:
                            ma_status = "Uptrend"
                            ma_color = "lightgreen"
                        elif current_price < sma_20 < sma_50:
                            ma_status = "Strong Downtrend"
                            ma_color = "red"
                        elif current_price < sma_20:
                            ma_status = "Downtrend"
                            ma_color = "pink"
                        else:
                            ma_status = "Consolidating"
                            ma_color = "gray"
                        
                        # Calculate distance from price to SMA20
                        distance = (current_price / sma_20 - 1) * 100
                        
                        st.metric(
                            label="Trend Status",
                            value=ma_status,
                            delta=f"{distance:.2f}% from SMA20"
                        )
                        st.markdown(f"<span style='color:{ma_color};'>{ma_status}</span>", unsafe_allow_html=True)
                    else:
                        st.metric(
                            label="Trend Status",
                            value="N/A"
                        )
