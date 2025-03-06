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

# Import local modules
from data_utils import calculate_technical_indicators, generate_technical_signals
from models import (StockData, GeometricBrownianMotion, JumpDiffusionModel, 
                   HestonModel, VarianceGammaModel)

# Suppress warnings
warnings.filterwarnings('ignore')

def display_day_trader_section(ticker):
    """Display live trading chart with predictions."""
    st.header(f"Live Day Trading Analysis - {ticker}")
    
    # Initialize session state for tracking data
    if 'live_data' not in st.session_state:
        st.session_state.live_data = None
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = None
    if 'prediction_model' not in st.session_state:
        st.session_state.prediction_model = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'prediction_confidence' not in st.session_state:
        st.session_state.prediction_confidence = None
    if 'prediction_time' not in st.session_state:
        st.session_state.prediction_time = None
    if 'prediction_accuracy' not in st.session_state:
        st.session_state.prediction_accuracy = []
    if 'model_params' not in st.session_state:
        st.session_state.model_params = {}
    
    # Layout with two columns: main chart and controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main container for the chart
        chart_container = st.container()
    
    with col2:
        # Controls and settings
        display_trading_controls(ticker)
    
    # Fetch live data
    with chart_container:
        live_data = fetch_live_data(ticker)
        
        if live_data is not None and not live_data.empty:
            # Update session state
            st.session_state.live_data = live_data
            st.session_state.last_update_time = datetime.now()
            
            # Display main chart
            fig = create_live_chart(live_data, ticker)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display technical indicators
            display_technical_indicators(live_data)
        else:
            st.error(f"Failed to fetch live data for {ticker}. Please try again or select a different ticker.")
    
    # Update button at the bottom
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Refresh Data & Update Predictions", key="refresh_data", use_container_width=True):
            # Directly fetch new data without rerun to avoid errors
            new_data = fetch_live_data(ticker)
            if new_data is not None and not new_data.empty:
                st.session_state.live_data = new_data
                st.session_state.last_update_time = datetime.now()
                # Generate new predictions if we had previous prediction settings
                if 'model_params' in st.session_state and st.session_state.model_params:
                    params = st.session_state.model_params
                    generate_price_predictions(ticker, 
                                              params.get('model_type', 'Jump Diffusion'), 
                                              params.get('horizon', 15), 
                                              params.get('num_sims', 1000))
            # Use the regular reload mechanism
            st.success("Data and predictions updated!")
            time.sleep(1)  # Give a moment for the success message to show
            st.empty()  # Clear the success message

def fetch_live_data(ticker):
    """Fetch live intraday data for the selected ticker."""
    try:
        # Create a progress message
        with st.spinner(f"Fetching live data for {ticker}..."):
            # Use yfinance to get intraday data
            stock = yf.Ticker(ticker)
            
            # Get 1-minute data for the most recent trading day
            # For live trading, we'll use the most granular data available
            intraday_data = stock.history(period="1d", interval="1m")
            
            if len(intraday_data) > 0:
                # If market is closed, we might get data from the last trading day
                # Convert index to US Eastern time for market hours
                eastern = pytz.timezone('US/Eastern')
                intraday_data.index = intraday_data.index.tz_convert(eastern)
                
                # Also fetch some historical daily data for model calibration
                daily_data = stock.history(period="3mo")
                
                # Store both in session state for model calibration
                st.session_state.intraday_data = intraday_data
                st.session_state.daily_data = daily_data
                
                # Get current market price
                current_price = intraday_data['Close'].iloc[-1]
                st.session_state.current_price = current_price
                
                return intraday_data
            else:
                st.error("No intraday data available. Market may be closed.")
                return None
    except Exception as e:
        st.error(f"Error fetching live data: {e}")
        return None

def create_live_chart(data, ticker):
    """Create an interactive live chart with candlesticks and predictions."""
    # Create subplots with price and volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, row_heights=[0.7, 0.3],
                       subplot_titles=("Price", "Volume"))
    
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
    data['SMA_9'] = data['Close'].rolling(window=9).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['SMA_9'],
            line=dict(color='blue', width=1),
            name="SMA 9"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            line=dict(color='orange', width=1),
            name="SMA 20"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['EMA_9'],
            line=dict(color='purple', width=1, dash='dash'),
            name="EMA 9"
        ),
        row=1, col=1
    )
    
    # Add predictions if available
    if st.session_state.predictions is not None and len(st.session_state.predictions) > 0:
        # Extract the last actual timestamp
        last_time = data.index[-1]
        
        # Generate future timestamps for predictions
        future_times = [last_time + timedelta(minutes=i+1) for i in range(len(st.session_state.predictions))]
        
        # Add prediction line
        fig.add_trace(
            go.Scatter(
                x=[last_time] + future_times,
                y=[data['Close'].iloc[-1]] + st.session_state.predictions,
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
        title=f"{ticker} Live Price with Predictions",
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
    
    return fig

def display_trading_controls(ticker):
    """Display live trading controls and indicators panel."""
    # Current price display
    if 'current_price' in st.session_state:
        current_price = st.session_state.current_price
        
        # Get daily data for percentage change calculation
        if 'daily_data' in st.session_state and len(st.session_state.daily_data) > 0:
            daily_data = st.session_state.daily_data
            prev_close = daily_data['Close'].iloc[-2] if len(daily_data) > 1 else daily_data['Close'].iloc[0]
            change = (current_price / prev_close - 1) * 100
            
            # Display current price with change
            st.metric(
                label="Current Price",
                value=f"${current_price:.2f}",
                delta=f"{change:.2f}%"
            )
        else:
            st.metric(
                label="Current Price",
                value=f"${current_price:.2f}"
            )
    
    # Last update time
    if st.session_state.last_update_time:
        st.markdown(f"""
        <div style='border: 1px solid #e6f3ff; padding: 5px; background-color: #2f2f2f; margin-bottom: 10px; text-align: center;'>
        Last updated: {st.session_state.last_update_time.strftime('%H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction controls
    st.markdown("<div style='border: 1px solid #e6f3ff; padding: 10px; background-color: #2f2f2f; margin-top: 10px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 16px; color: #e6f3ff; margin-bottom: 10px;'>Prediction Settings</div>", unsafe_allow_html=True)
    
    # Model selection
    model_type = st.selectbox(
        "Prediction Model",
        ["Jump Diffusion", "Heston", "Variance Gamma", "GBM"]
    )
    
    # Prediction horizon
    horizon = st.slider("Prediction Steps (minutes)", min_value=5, max_value=30, value=15, step=5)
    
    # Number of simulations
    num_sims = st.slider("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)
    
    # Store model parameters in session state
    st.session_state.model_params = {
        'model_type': model_type,
        'horizon': horizon,
        'num_sims': num_sims
    }
    
    # Button to generate prediction
    if st.button("Generate Prediction", key="generate_prediction", use_container_width=True):
        with st.spinner("Generating price predictions..."):
            generate_price_predictions(ticker, model_type, horizon, num_sims)
    
    # Show prediction details if available
    if st.session_state.predictions is not None and st.session_state.prediction_time is not None:
        st.markdown(f"""
        <div style='margin-top: 10px; font-size: 14px; color: #e6f3ff;'>
        Prediction generated at {st.session_state.prediction_time.strftime('%H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)
        
        # Get prediction direction
        current_price = st.session_state.current_price if 'current_price' in st.session_state else 0
        last_predicted = st.session_state.predictions[-1] if st.session_state.predictions else 0
        
        if current_price > 0 and last_predicted > 0:
            direction = "UP" if last_predicted > current_price else "DOWN"
            direction_color = "green" if direction == "UP" else "red"
            change = (last_predicted / current_price - 1) * 100
            
            st.markdown(f"""
            <div style='margin-top: 5px; font-size: 18px; color: {direction_color}; font-weight: bold; text-align: center;'>
            Predicted Direction: {direction} ({change:.2f}%)
            </div>
            """, unsafe_allow_html=True)
            
        # Display confidence level if available
        if st.session_state.prediction_confidence is not None:
            confidence = st.session_state.prediction_confidence
            conf_level = confidence['level']
            conf_pct = confidence['percentage']
            
            # Color based on confidence level
            if conf_level == "High":
                conf_color = "#4CAF50"  # Green
            elif conf_level == "Moderate":
                conf_color = "#FFC107"  # Amber
            elif conf_level == "Low":
                conf_color = "#FF9800"  # Orange
            else:
                conf_color = "#F44336"  # Red
            
            # Display confidence
            st.markdown(f"""
            <div style='margin-top: 10px; text-align: center;'>
                <div style='font-size: 14px; color: #e6f3ff; margin-bottom: 5px;'>Prediction Confidence</div>
                <div style='font-size: 16px; color: {conf_color}; font-weight: bold;'>{conf_level} ({conf_pct}%)</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Visual confidence bar
            st.progress(conf_pct/100, text="")
            
            with st.expander("Confidence Factors", expanded=False):
                factors = confidence['factors']
                st.markdown("""
                Confidence is calculated based on:
                - **Time Horizon**: Shorter predictions are more accurate
                - **Model Type**: Some models perform better for day trading
                - **Volatility**: Higher volatility reduces prediction confidence
                - **Data Quality**: More data points improve prediction quality
                """)
                
                # Create a table of confidence factors
                factor_df = pd.DataFrame({
                    'Factor': ['Time Horizon', 'Model', 'Volatility', 'Data Quality'],
                    'Score': [
                        f"{factors['time_horizon']*100:.0f}%",
                        f"{factors['model']*100:.0f}%",
                        f"{factors['volatility']*100:.0f}%",
                        f"{factors['data_quality']*100:.0f}%"
                    ]
                })
                st.dataframe(factor_df, hide_index=True)
    
    # Prediction accuracy if available
    if st.session_state.prediction_accuracy and len(st.session_state.prediction_accuracy) > 0:
        correct = sum(1 for x in st.session_state.prediction_accuracy if x)
        total = len(st.session_state.prediction_accuracy)
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        st.markdown(f"""
        <div style='margin-top: 5px; font-size: 14px; color: #e6f3ff;'>
        Prediction Accuracy: {accuracy:.1f}% ({correct}/{total} correct)
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Technical indicators summary
    display_indicators_summary()

def calculate_prediction_confidence(model_type, horizon, volatility, data_length):
    """Calculate a confidence score for the prediction."""
    # Base confidence level based on timeframe
    if horizon <= 5:
        time_confidence = 0.75  # 5 minutes or less
    elif horizon <= 15:
        time_confidence = 0.60  # 5-15 minutes
    else:
        time_confidence = 0.45  # Over 15 minutes
    
    # Model confidence
    model_confidence = {
        "Jump Diffusion": 0.70,  # Good for capturing sudden moves
        "Heston": 0.65,          # Good for varying volatility
        "Variance Gamma": 0.68,  # Good for fat tails
        "GBM": 0.55              # Simplest model
    }.get(model_type, 0.60)
    
    # Volatility factor - lower confidence with higher volatility
    vol_factor = max(0.4, min(1.0, 1.0 - (volatility - 0.1) * 2))
    
    # Data quality factor - more data points increases confidence
    data_factor = min(1.0, data_length / 120)  # Normalize to 2 hours of minute data
    
    # Combine factors
    confidence = (time_confidence * 0.4 + 
                 model_confidence * 0.3 + 
                 vol_factor * 0.2 + 
                 data_factor * 0.1)
    
    # Scale to percentage
    confidence_pct = min(95, max(30, round(confidence * 100)))
    
    # Confidence level text
    if confidence_pct >= 80:
        level = "High"
    elif confidence_pct >= 60:
        level = "Moderate"
    elif confidence_pct >= 40:
        level = "Low"
    else:
        level = "Very Low"
    
    return {
        "percentage": confidence_pct,
        "level": level,
        "factors": {
            "time_horizon": time_confidence,
            "model": model_confidence,
            "volatility": vol_factor,
            "data_quality": data_factor
        }
    }

def generate_price_predictions(ticker, model_type, horizon, num_sims):
    """Generate price predictions using selected model."""
    try:
        # Check if we have data to work with
        if 'intraday_data' not in st.session_state or st.session_state.intraday_data is None:
            st.error("No intraday data available for prediction.")
            return
        
        # Get current price
        current_price = st.session_state.current_price if 'current_price' in st.session_state else 0
        
        if current_price == 0:
            st.error("Invalid current price for prediction.")
            return
        
        # Create a StockData object for the model
        alpha_vantage_key = ""  # Not needed for this prediction
        fred_api_key = ""       # Not needed for this prediction
        
        stock_data = StockData(ticker, alpha_vantage_key, fred_api_key)
        stock_data.price = current_price
        
        # Calculate volatility from intraday data
        intraday_returns = st.session_state.intraday_data['Close'].pct_change().dropna()
        intraday_volatility = intraday_returns.std() * np.sqrt(252 * 6.5 * 60)  # Annualized from 1-minute data
        stock_data.volatility = intraday_volatility
        
        # Use a default risk-free rate
        stock_data.risk_free_rate = 0.04
        
        # Set historical data for calibration
        stock_data.historical_data = st.session_state.daily_data
        stock_data.returns = stock_data.historical_data['Close'].pct_change().dropna()
        
        # Select model based on user choice
        if model_type == "Jump Diffusion":
            model = JumpDiffusionModel(stock_data)
        elif model_type == "Heston":
            model = HestonModel(stock_data)
        elif model_type == "Variance Gamma":
            model = VarianceGammaModel(stock_data)
        else:  # Default to GBM
            model = GeometricBrownianMotion(stock_data)
        
        # Calibrate the model
        model.calibrate()
        
        # Run simulation for short time horizon
        T = horizon / (252 * 6.5 * 60)  # Convert minutes to years (assuming 252 trading days, 6.5 hours per day)
        dt = T / horizon  # Time step size
        M = num_sims  # Number of simulations
        
        # Execute the simulation
        simulation_result = model.simulate(T, dt, M, current_price * 1.01)  # Target price doesn't matter here
        
        # Extract predictions for each minute
        predicted_prices = []
        upper_bounds = []
        lower_bounds = []
        
        # Check if we have full paths in the result
        if 'paths' in simulation_result:
            # Use the full paths directly
            paths = simulation_result['paths']
            
            # Calculate average price at each time step
            for i in range(1, horizon + 1):
                if i < paths.shape[1]:
                    step_prices = paths[:, i]
                    avg_price = np.mean(step_prices)
                    predicted_prices.append(avg_price)
                    
                    # Calculate confidence intervals (90%)
                    upper_bounds.append(np.percentile(step_prices, 95))
                    lower_bounds.append(np.percentile(step_prices, 5))
        else:
            # No full paths available, generate simple prediction
            # Use a modified random walk based on current model parameters
            predicted_price = current_price
            for i in range(horizon):
                drift = model.r * dt
                vol = model.sigma * np.sqrt(dt)
                
                # Simulate many paths and take the average
                step_prices = []
                for j in range(num_sims):
                    random_shock = np.random.normal(0, 1)
                    next_price = predicted_price * np.exp(drift + vol * random_shock)
                    step_prices.append(next_price)
                
                predicted_price = np.mean(step_prices)
                predicted_prices.append(predicted_price)
                
                # Calculate confidence intervals
                upper_bounds.append(np.percentile(step_prices, 95))
                lower_bounds.append(np.percentile(step_prices, 5))
        
        # Calculate prediction confidence
        confidence = calculate_prediction_confidence(
            model_type, 
            horizon, 
            intraday_volatility,
            len(st.session_state.intraday_data)
        )
        
        # Store predictions in session state
        st.session_state.predictions = predicted_prices
        st.session_state.prediction_upper = upper_bounds
        st.session_state.prediction_lower = lower_bounds
        st.session_state.prediction_time = datetime.now()
        st.session_state.prediction_model = model_type
        st.session_state.prediction_confidence = confidence
        
        st.success(f"Generated {horizon}-minute prediction using {model_type} model with {confidence['level']} confidence ({confidence['percentage']}%)")
        
    except Exception as e:
        st.error(f"Error generating predictions: {e}")

def display_technical_indicators(data):
    """Calculate and display key technical indicators."""
    # Ensure we have enough data
    if len(data) < 20:
        st.warning("Insufficient data for technical indicators.")
        return
    
    # Calculate technical indicators
    indicators = calculate_technical_indicators(data)
    
    # Get latest values
    latest = indicators.iloc[-1]
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # RSI
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
            st.markdown(f"<span style='color:{rsi_color};font-size:14px;'>{rsi_status}</span>", unsafe_allow_html=True)
    
    with col2:
        # MACD
        if all(k in latest for k in ['MACD', 'MACD_Signal']):
            macd_value = latest['MACD']
            signal_value = latest['MACD_Signal']
            macd_diff = macd_value - signal_value
            macd_status = "Bullish" if macd_diff > 0 else "Bearish"
            macd_color = "green" if macd_diff > 0 else "red"
            
            st.metric(
                label="MACD",
                value=f"{macd_value:.4f}",
                delta=f"Signal: {signal_value:.4f}",
                delta_color="off"
            )
            st.markdown(f"<span style='color:{macd_color};font-size:14px;'>{macd_status} ({macd_diff:.4f})</span>", unsafe_allow_html=True)
    
    with col3:
        # Bollinger Bands
        if all(k in latest for k in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            price = latest['Close']
            bb_upper = latest['BB_Upper']
            bb_lower = latest['BB_Lower']
            
            # Calculate %B value
            pct_b = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5
            bb_status = "Upper Band" if pct_b > 0.8 else "Lower Band" if pct_b < 0.2 else "Middle Band"
            bb_color = "red" if pct_b > 0.8 else "green" if pct_b < 0.2 else "gray"
            
            st.metric(
                label="Bollinger Bands",
                value=f"${price:.2f}",
                delta=f"B%: {pct_b*100:.1f}%",
                delta_color="off"
            )
            st.markdown(f"<span style='color:{bb_color};font-size:14px;'>{bb_status}</span>", unsafe_allow_html=True)

def display_indicators_summary():
    """Display a summary of technical signals."""
    if 'intraday_data' not in st.session_state or st.session_state.intraday_data is None:
        return
    
    data = st.session_state.intraday_data
    
    # Calculate technical indicators
    if len(data) >= 20:
        indicators = calculate_technical_indicators(data)
        signals = generate_technical_signals(indicators)
        
        st.markdown("<div style='border: 1px solid #e6f3ff; padding: 10px; background-color: #2f2f2f; margin-top: 10px;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 16px; color: #e6f3ff; margin-bottom: 10px;'>Technical Signals</div>", unsafe_allow_html=True)
        
        # Display summary signal first
        if "Summary" in signals:
            for signal in signals["Summary"]:
                signal_color = "green" if signal['direction'] == "bullish" else "red" if signal['direction'] == "bearish" else "gray"
                st.markdown(f"<div style='font-size: 18px; color: {signal_color}; text-align: center; margin-bottom: 5px;'>{signal['description']}</div>", unsafe_allow_html=True)
        
        # Expandable section for detailed signals
        with st.expander("View Detailed Signals", expanded=False):
            # Display signals by category
            for category in ["Moving Averages", "Oscillators", "Trend Indicators"]:
                if category in signals:
                    st.markdown(f"<div style='font-size: 14px; color: #e6f3ff; margin-top: 5px;'>{category}</div>", unsafe_allow_html=True)
                    for signal in signals[category]:
                        signal_color = "green" if signal['direction'] == "bullish" else "red" if signal['direction'] == "bearish" else "gray"
                        st.markdown(f"<div style='font-size: 12px; color: {signal_color};'>• {signal['description']}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
