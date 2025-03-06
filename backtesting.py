import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import warnings

# Import local modules
from models import (
    StockData, GeometricBrownianMotion, JumpDiffusionModel, 
    HestonModel, GARCHModel, VarianceGammaModel, QuasiMonteCarloModel,
    RegimeSwitchingModel, NeuralSDEModel, StockModelEnsemble
)

# Suppress warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    """Engine for backtesting stock price prediction models."""
    
    def __init__(self, ticker, start_date, end_date, alpha_vantage_key="", fred_api_key=""):
        """Initialize backtesting engine."""
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.alpha_vantage_key = alpha_vantage_key
        self.fred_api_key = fred_api_key
        self.historical_data = None
        self.test_windows = []
        self.model_results = {}
        self.ensemble_results = {}
    
    def fetch_data(self):
        """Fetch historical data for the specified period."""
        try:
            stock = yf.Ticker(self.ticker)
            data = stock.history(start=self.start_date, end=self.end_date)
            if len(data) > 0:
                self.historical_data = data
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Error fetching historical data: {e}")
            return False
    
    def fetch_intraday_data(self, days=60, interval="1h"):
        """Fetch intraday data for short-term backtesting."""
        try:
            stock = yf.Ticker(self.ticker)
            # Use period instead of start/end for intraday data
            data = stock.history(period=f"{days}d", interval=interval)
            if len(data) > 0:
                self.historical_data = data
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Error fetching intraday data: {e}")
            return False
    
    def create_test_windows(self, window_size=30, step_size=7, min_train_size=252):
        """Create sliding windows for walk-forward testing.
        
        Args:
            window_size: Size of test window in days
            step_size: Days to move forward for next window
            min_train_size: Minimum size of training data in days
        """
        if self.historical_data is None:
            st.error("No historical data available. Please fetch data first.")
            return False
        
        # Reset test windows
        self.test_windows = []
        
        # Get all available dates
        all_dates = self.historical_data.index
        
        # Check if we have enough data
        if len(all_dates) < min_train_size + window_size:
            st.warning(f"Insufficient data for backtesting with minimum {min_train_size} days training and {window_size} days testing.")
            return False
        
        # Create windows
        start_idx = min_train_size
        while start_idx + window_size <= len(all_dates):
            train_data = self.historical_data.iloc[:start_idx]
            test_data = self.historical_data.iloc[start_idx:start_idx + window_size]
            
            self.test_windows.append({
                'train_start': all_dates[0],
                'train_end': all_dates[start_idx - 1],
                'test_start': all_dates[start_idx],
                'test_end': all_dates[min(start_idx + window_size - 1, len(all_dates) - 1)],
                'train_data': train_data,
                'test_data': test_data
            })
            
            start_idx += step_size
        
        return True
    
    def create_intraday_test_windows(self, window_size_hours=8, step_size_hours=24, min_train_size_hours=40):
        """Create sliding windows for intraday backtesting."""
        if self.historical_data is None:
            st.error("No intraday data available. Please fetch data first.")
            return False
        
        # Reset test windows
        self.test_windows = []
        
        # Get all available timestamps
        all_timestamps = self.historical_data.index
        
        # Check if we have enough data
        if len(all_timestamps) < min_train_size_hours + window_size_hours:
            st.warning(f"Insufficient data for backtesting with minimum {min_train_size_hours} hours training and {window_size_hours} hours testing.")
            return False
        
        # Create windows based on trading hours
        # For hourly data, each step is 1 hour
        trading_hours_per_day = 6.5  # NYSE is open 9:30 AM to 4:00 PM (6.5 hours)
        
        # Convert to indices
        window_size_idx = window_size_hours
        step_size_idx = step_size_hours
        min_train_idx = min_train_size_hours
        
        start_idx = min_train_idx
        while start_idx + window_size_idx <= len(all_timestamps):
            train_data = self.historical_data.iloc[:start_idx]
            test_data = self.historical_data.iloc[start_idx:start_idx + window_size_idx]
            
            self.test_windows.append({
                'train_start': all_timestamps[0],
                'train_end': all_timestamps[start_idx - 1],
                'test_start': all_timestamps[start_idx],
                'test_end': all_timestamps[min(start_idx + window_size_idx - 1, len(all_timestamps) - 1)],
                'train_data': train_data,
                'test_data': test_data
            })
            
            start_idx += step_size_idx
        
        return True
    
    def run_model_backtest(self, model_class, model_params=None, target_price_pct=1.05):
        """Run backtest for a specific model across all test windows."""
        if not self.test_windows:
            st.error("No test windows defined. Please create test windows first.")
            return False
        
        # Store results for this model
        model_name = model_class.__name__
        self.model_results[model_name] = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run backtest for each window
        for i, window in enumerate(self.test_windows):
            status_text.text(f"Testing {model_name} - Window {i+1}/{len(self.test_windows)}")
            
            # Create StockData object for this window
            stock_data = StockData(self.ticker, self.alpha_vantage_key, self.fred_api_key)
            
            # Use the last price from training data
            stock_data.price = window['train_data']['Close'].iloc[-1]
            
            # Calculate volatility from training data
            returns = window['train_data']['Close'].pct_change().dropna()
            stock_data.volatility = returns.std() * np.sqrt(252)
            
            # Set default risk-free rate
            stock_data.risk_free_rate = 0.04
            
            # Add historical data for model calibration
            stock_data.historical_data = window['train_data']
            stock_data.returns = returns
            
            # Initialize model
            if model_params:
                model = model_class(stock_data, **model_params)
            else:
                model = model_class(stock_data)
            
            # Calibrate model
            model.calibrate()
            
            # Get test data
            test_dates = window['test_data'].index
            test_prices = window['test_data']['Close']
            
            # Calculate target prices for each day (5% above start price)
            start_price = stock_data.price
            target_price = start_price * target_price_pct
            
            # Run simulation for the test period
            simulation_years = (test_dates[-1] - test_dates[0]).days / 365
            
            if simulation_years < 0.01:  # For very short periods, use a minimum
                simulation_years = 0.01
                
            # Run simulation
            result = model.simulate(
                T=simulation_years,
                dt=simulation_years / len(test_dates) if len(test_dates) > 0 else 0.01,
                M=1000,  # Number of simulations
                target_price=target_price
            )
            
            # Calculate metrics
            # 1. Direction accuracy - did the model predict the right direction?
            actual_direction = 1 if test_prices.iloc[-1] > start_price else 0
            predicted_direction = 1 if result['mean_price'] > start_price else 0
            direction_correct = actual_direction == predicted_direction
            
            # 2. Price accuracy - how close was the predicted final price?
            price_error = abs(result['mean_price'] - test_prices.iloc[-1]) / test_prices.iloc[-1] * 100
            
            # 3. Probability accuracy - was the target reached?
            target_reached = 1 if max(test_prices) >= target_price else 0
            probability_error = abs(result['max_probability'] / 100 - target_reached) * 100
            
            # Store results
            window_result = {
                'model': model_name,
                'train_start': window['train_start'],
                'train_end': window['train_end'],
                'test_start': window['test_start'],
                'test_end': window['test_end'],
                'start_price': start_price,
                'final_price': test_prices.iloc[-1],
                'predicted_final': result['mean_price'],
                'actual_return': (test_prices.iloc[-1] / start_price - 1) * 100,
                'predicted_return': (result['mean_price'] / start_price - 1) * 100,
                'target_price': target_price,
                'target_reached': target_reached,
                'predicted_probability': result['max_probability'],
                'direction_correct': direction_correct,
                'price_error_pct': price_error,
                'probability_error_pct': probability_error,
                'confidence_interval': result['confidence_interval']
            }
            
            self.model_results[model_name].append(window_result)
            
            # Update progress
            progress_bar.progress((i + 1) / len(self.test_windows))
        
        status_text.text(f"Completed backtest for {model_name}")
        progress_bar.empty()
        
        return True
    
    def run_ensemble_backtest(self, model_classes, target_price_pct=1.05):
        """Run backtest using an ensemble of models."""
        if not self.test_windows:
            st.error("No test windows defined. Please create test windows first.")
            return False
        
        # Store results for the ensemble
        self.ensemble_results = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run backtest for each window
        for i, window in enumerate(self.test_windows):
            status_text.text(f"Testing Ensemble - Window {i+1}/{len(self.test_windows)}")
            
            # Create StockData object for this window
            stock_data = StockData(self.ticker, self.alpha_vantage_key, self.fred_api_key)
            
            # Use the last price from training data
            stock_data.price = window['train_data']['Close'].iloc[-1]
            
            # Calculate volatility from training data
            returns = window['train_data']['Close'].pct_change().dropna()
            stock_data.volatility = returns.std() * np.sqrt(252)
            
            # Set default risk-free rate
            stock_data.risk_free_rate = 0.04
            
            # Add historical data for model calibration
            stock_data.historical_data = window['train_data']
            stock_data.returns = returns
            
            # Create ensemble
            ensemble = StockModelEnsemble(stock_data)
            
            # Add all specified models
            for model_class in model_classes:
                ensemble.add_model(model_class(stock_data))
            
            # Calibrate all models
            ensemble.calibrate_all()
            
            # Get test data
            test_dates = window['test_data'].index
            test_prices = window['test_data']['Close']
            
            # Calculate target price (5% above start price)
            start_price = stock_data.price
            target_price = start_price * target_price_pct
            
            # Run simulation for the test period
            simulation_years = (test_dates[-1] - test_dates[0]).days / 365
            
            if simulation_years < 0.01:  # For very short periods, use a minimum
                simulation_years = 0.01
            
            # Run simulations for all models
            ensemble.run_all_simulations(
                T=simulation_years,
                dt=simulation_years / len(test_dates) if len(test_dates) > 0 else 0.01,
                M=1000,  # Number of simulations
                target_price=target_price
            )
            
            # Compute ensemble forecast
            ensemble_result = ensemble.compute_ensemble_forecast()
            
            # Calculate metrics
            # 1. Direction accuracy - did the model predict the right direction?
            actual_direction = 1 if test_prices.iloc[-1] > start_price else 0
            predicted_direction = 1 if ensemble_result['mean_price'] > start_price else 0
            direction_correct = actual_direction == predicted_direction
            
            # 2. Price accuracy - how close was the predicted final price?
            price_error = abs(ensemble_result['mean_price'] - test_prices.iloc[-1]) / test_prices.iloc[-1] * 100
            
            # 3. Probability accuracy - was the target reached?
            target_reached = 1 if max(test_prices) >= target_price else 0
            probability_error = abs(ensemble_result['max_probability'] / 100 - target_reached) * 100
            
            # Store results
            window_result = {
                'model': 'Ensemble',
                'train_start': window['train_start'],
                'train_end': window['train_end'],
                'test_start': window['test_start'],
                'test_end': window['test_end'],
                'start_price': start_price,
                'final_price': test_prices.iloc[-1],
                'predicted_final': ensemble_result['mean_price'],
                'actual_return': (test_prices.iloc[-1] / start_price - 1) * 100,
                'predicted_return': (ensemble_result['mean_price'] / start_price - 1) * 100,
                'target_price': target_price,
                'target_reached': target_reached,
                'predicted_probability': ensemble_result['max_probability'],
                'direction_correct': direction_correct,
                'price_error_pct': price_error,
                'probability_error_pct': probability_error,
                'model_weights': ensemble_result['model_ranking'],
                'confidence_interval': ensemble_result['confidence_interval']
            }
            
            self.ensemble_results.append(window_result)
            
            # Update progress
            progress_bar.progress((i + 1) / len(self.test_windows))
        
        status_text.text("Completed ensemble backtest")
        progress_bar.empty()
        
        return True
    
    def get_summary_metrics(self):
        """Calculate summary metrics for all models."""
        metrics = {}
        
        # Process individual model results
        for model_name, results in self.model_results.items():
            if not results:
                continue
                
            df = pd.DataFrame(results)
            
            metrics[model_name] = {
                'direction_accuracy': df['direction_correct'].mean() * 100,
                'avg_price_error': df['price_error_pct'].mean(),
                'avg_probability_error': df['probability_error_pct'].mean(),
                'avg_predicted_return': df['predicted_return'].mean(),
                'avg_actual_return': df['actual_return'].mean(),
                'return_bias': df['predicted_return'].mean() - df['actual_return'].mean(),
                'windows_tested': len(df)
            }
        
        # Process ensemble results
        if self.ensemble_results:
            df = pd.DataFrame(self.ensemble_results)
            
            metrics['Ensemble'] = {
                'direction_accuracy': df['direction_correct'].mean() * 100,
                'avg_price_error': df['price_error_pct'].mean(),
                'avg_probability_error': df['probability_error_pct'].mean(),
                'avg_predicted_return': df['predicted_return'].mean(),
                'avg_actual_return': df['actual_return'].mean(),
                'return_bias': df['predicted_return'].mean() - df['actual_return'].mean(),
                'windows_tested': len(df)
            }
        
        return metrics

def clear_backtest_state():
    """Clear backtesting state when ticker changes."""
    if 'backtest_engine' in st.session_state:
        del st.session_state.backtest_engine
    if 'backtest_results' in st.session_state:
        del st.session_state.backtest_results
    if 'backtest_ticker' in st.session_state:
        del st.session_state.backtest_ticker

def display_backtesting_section(ticker):
    """Display backtesting section in the app."""
    st.header("Model Backtesting")
    
    # Check if ticker has changed and clear results if needed
    if 'backtest_ticker' in st.session_state and st.session_state.backtest_ticker != ticker:
        st.info(f"Ticker changed from {st.session_state.backtest_ticker} to {ticker}. Backtesting results have been cleared.")
        clear_backtest_state()
    
    # Initialize session state
    if 'backtest_engine' not in st.session_state:
        st.session_state.backtest_engine = None
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    
    # Store current ticker in session state
    st.session_state.backtest_ticker = ticker
    
    # Sidebar for backtest settings
    with st.sidebar:
        st.markdown("<h3 style='color: #e6f3ff;'>Backtest Settings</h3>", unsafe_allow_html=True)
        
        timeframe = st.selectbox(
            "Timeframe",
            ["Daily", "Hourly", "Intraday (15min)"]
        )
        
        # Default dates based on timeframe
        today = datetime.now()
        if timeframe == "Daily":
            default_start = today - timedelta(days=365*3)  # 3 years
            default_end = today
        elif timeframe == "Hourly":
            default_start = today - timedelta(days=60)  # 60 days
            default_end = today
        else:  # Intraday
            default_start = today - timedelta(days=30)  # 30 days
            default_end = today
        
        # Date range selection
        start_date = st.date_input(
            "Start Date",
            value=default_start.date()
        )
        
        end_date = st.date_input(
            "End Date",
            value=default_end.date()
        )
        
        # Test window settings
        if timeframe == "Daily":
            window_size = st.slider("Test Window Size (days)", min_value=7, max_value=90, value=30, step=1)
            step_size = st.slider("Step Size (days)", min_value=1, max_value=30, value=7, step=1)
            min_train_size = st.slider("Min Training Size (days)", min_value=60, max_value=504, value=252, step=30)
        elif timeframe == "Hourly":
            window_size = st.slider("Test Window Size (hours)", min_value=4, max_value=48, value=8, step=4)
            step_size = st.slider("Step Size (hours)", min_value=4, max_value=24, value=8, step=4)
            min_train_size = st.slider("Min Training Size (hours)", min_value=24, max_value=200, value=40, step=8)
        else:  # Intraday
            window_size = st.slider("Test Window Size (15min periods)", min_value=4, max_value=48, value=16, step=4)
            step_size = st.slider("Step Size (15min periods)", min_value=4, max_value=24, value=8, step=4)
            min_train_size = st.slider("Min Training Size (15min periods)", min_value=16, max_value=200, value=32, step=8)
        
        # Target price percentage
        target_price_pct = st.slider("Target Price % (above start)", min_value=1.01, max_value=1.20, value=1.05, step=0.01)
        
        # Model selection
        st.markdown("<h3 style='color: #e6f3ff;'>Models to Test</h3>", unsafe_allow_html=True)
        
        test_gbm = st.checkbox("Geometric Brownian Motion", value=True)
        test_jump = st.checkbox("Jump Diffusion", value=True)
        test_heston = st.checkbox("Heston Stochastic Volatility", value=True)
        test_vg = st.checkbox("Variance Gamma", value=True)
        test_regime = st.checkbox("Regime Switching", value=True)
        test_qmc = st.checkbox("Quasi Monte Carlo", value=True)
        test_ensemble = st.checkbox("Ensemble (all models)", value=True)
        
        # Button to run backtest
        if st.button("Run Backtest", use_container_width=True):
            # Clear previous results when running new backtest
            if 'backtest_engine' in st.session_state:
                del st.session_state.backtest_engine
            if 'backtest_results' in st.session_state:
                del st.session_state.backtest_results
                
            with st.spinner("Fetching historical data..."):
                # Create backtest engine
                backtest_engine = BacktestEngine(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Fetch data based on timeframe
                success = False
                if timeframe == "Daily":
                    success = backtest_engine.fetch_data()
                    if success:
                        success = backtest_engine.create_test_windows(
                            window_size=window_size,
                            step_size=step_size,
                            min_train_size=min_train_size
                        )
                elif timeframe == "Hourly":
                    success = backtest_engine.fetch_intraday_data(days=(end_date - start_date).days, interval="1h")
                    if success:
                        success = backtest_engine.create_intraday_test_windows(
                            window_size_hours=window_size,
                            step_size_hours=step_size,
                            min_train_size_hours=min_train_size
                        )
                else:  # Intraday
                    success = backtest_engine.fetch_intraday_data(days=(end_date - start_date).days, interval="15m")
                    if success:
                        success = backtest_engine.create_intraday_test_windows(
                            window_size_hours=window_size,
                            step_size_hours=step_size,
                            min_train_size_hours=min_train_size
                        )
                
                if not success:
                    st.error("Failed to prepare data for backtesting.")
                    return
                
                # Store in session state
                st.session_state.backtest_engine = backtest_engine
                
                # Run selected model backtests
                if test_gbm:
                    backtest_engine.run_model_backtest(GeometricBrownianMotion, target_price_pct=target_price_pct)
                
                if test_jump:
                    backtest_engine.run_model_backtest(JumpDiffusionModel, target_price_pct=target_price_pct)
                
                if test_heston:
                    backtest_engine.run_model_backtest(HestonModel, target_price_pct=target_price_pct)
                
                if test_vg:
                    backtest_engine.run_model_backtest(VarianceGammaModel, target_price_pct=target_price_pct)
                
                if test_regime:
                    backtest_engine.run_model_backtest(RegimeSwitchingModel, target_price_pct=target_price_pct)
                
                if test_qmc:
                    backtest_engine.run_model_backtest(QuasiMonteCarloModel, target_price_pct=target_price_pct)
                
                # Run ensemble if requested and we have at least 2 models
                if test_ensemble:
                    ensemble_models = []
                    if test_gbm:
                        ensemble_models.append(GeometricBrownianMotion)
                    if test_jump:
                        ensemble_models.append(JumpDiffusionModel)
                    if test_heston:
                        ensemble_models.append(HestonModel)
                    if test_vg:
                        ensemble_models.append(VarianceGammaModel)
                    if test_regime:
                        ensemble_models.append(RegimeSwitchingModel)
                    if test_qmc:
                        ensemble_models.append(QuasiMonteCarloModel)
                    
                    if len(ensemble_models) >= 2:
                        backtest_engine.run_ensemble_backtest(ensemble_models, target_price_pct=target_price_pct)
                
                # Calculate summary metrics
                st.session_state.backtest_results = backtest_engine.get_summary_metrics()
                
                st.success("Backtest completed!")
                # Force rerun to refresh the UI
                st.rerun()
    
    # Main content area - display backtest results
    display_backtest_results(ticker)

def display_backtest_results(ticker):
    """Display the backtest results in the main content area."""
    # Verify we're displaying results for the correct ticker
    if 'backtest_ticker' in st.session_state and st.session_state.backtest_ticker != ticker:
        st.warning(f"The backtest results are for {st.session_state.backtest_ticker}, not {ticker}. Please run a new backtest for {ticker}.")
        if st.button("Clear Backtest Results"):
            clear_backtest_state()
            st.rerun()
        return
    
    if not st.session_state.backtest_engine or not st.session_state.backtest_results:
        st.info("Run a backtest to see results here.")
        st.markdown("""
        ### How to Use Backtesting
        
        1. Select a **timeframe** (Daily, Hourly, or Intraday)
        2. Choose the **date range** to test
        3. Configure the **test window** size and step
        4. Select which **models** to test
        5. Click **Run Backtest** to start
        
        The backtest will show how well different models predict:
        - Direction (up/down)
        - Final price
        - Probability of reaching target
        
        Results will help you understand which models perform best for your specific trading timeframe and strategy.
        """)
        return
    
    # Get backtest engine and results from session state
    backtest_engine = st.session_state.backtest_engine
    metrics = st.session_state.backtest_results
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Summary Results", "Detailed Analysis", "Model Comparison"])
    
    # Tab 1: Summary Results
    with tab1:
        st.subheader("Model Performance Summary")
        
        # Create metrics table
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        
        # Format columns
        if not metrics_df.empty:
            metrics_df['direction_accuracy'] = metrics_df['direction_accuracy'].map("{:.1f}%".format)
            metrics_df['avg_price_error'] = metrics_df['avg_price_error'].map("{:.2f}%".format)
            metrics_df['avg_probability_error'] = metrics_df['avg_probability_error'].map("{:.2f}%".format)
            metrics_df['avg_predicted_return'] = metrics_df['avg_predicted_return'].map("{:.2f}%".format)
            metrics_df['avg_actual_return'] = metrics_df['avg_actual_return'].map("{:.2f}%".format)
            metrics_df['return_bias'] = metrics_df['return_bias'].map("{:.2f}%".format)
            
            # Rename columns for better readability
            metrics_df.columns = [
                'Direction Accuracy', 
                'Avg Price Error', 
                'Avg Probability Error',
                'Avg Predicted Return',
                'Avg Actual Return',
                'Return Bias',
                'Windows Tested'
            ]
            
            # Create a colorful bar chart for direction accuracy
            fig1 = px.bar(
                metrics_df.reset_index(),
                x='index',
                y='Direction Accuracy',
                title="Direction Prediction Accuracy by Model",
                labels={'index': 'Model', 'Direction Accuracy': 'Accuracy (%)'},
                color='Direction Accuracy',
                color_continuous_scale='blues',
                text='Direction Accuracy'
            )
            
            # Apply terminal styling
            fig1.update_layout(
                paper_bgcolor='#2f2f2f',
                plot_bgcolor='#2f2f2f',
                font=dict(color='#ffffff'),
                title_font=dict(color='#e6f3ff'),
                xaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff'),
                yaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff')
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Display metrics table
            st.dataframe(metrics_df, use_container_width=True)
            
            # Winner categories
            st.subheader("Best Performers")
            
            # Direction accuracy winner
            direction_winner = metrics_df['Direction Accuracy'].idxmax()
            direction_accuracy = metrics_df.loc[direction_winner, 'Direction Accuracy']
            
            # Price error winner (lowest error)
            price_error_winner = metrics_df['Avg Price Error'].astype(str).str.rstrip('%').astype(float).idxmin()
            price_error = metrics_df.loc[price_error_winner, 'Avg Price Error']
            
            # Most accurate return prediction (lowest bias)
            bias_winner = metrics_df['Return Bias'].astype(str).str.rstrip('%').astype(float).abs().idxmin()
            bias = metrics_df.loc[bias_winner, 'Return Bias']
            
            # Display winners
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Direction Predictor", direction_winner, direction_accuracy)
            
            with col2:
                st.metric("Most Price Accurate", price_error_winner, price_error)
            
            with col3:
                st.metric("Least Biased", bias_winner, bias)
    
    # Tab 2: Detailed Analysis
    with tab2:
        st.subheader("Detailed Backtest Analysis")
        
        # Select model to analyze
        available_models = list(backtest_engine.model_results.keys())
        if backtest_engine.ensemble_results:
            available_models.append("Ensemble")
        
        if not available_models:
            st.warning("No backtest results to display.")
            return
        
        selected_model = st.selectbox("Select Model to Analyze", available_models)
        
        # Get results for selected model
        if selected_model == "Ensemble":
            selected_results = backtest_engine.ensemble_results
        else:
            selected_results = backtest_engine.model_results.get(selected_model, [])
        
        if not selected_results:
            st.warning(f"No results available for {selected_model}.")
            return
        
        # Convert to DataFrame
        results_df = pd.DataFrame(selected_results)
        
        # Show results by time window
        st.subheader("Results by Test Window")
        
        # Line chart of predicted vs actual returns
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=results_df['test_end'],
            y=results_df['predicted_return'],
            mode='lines+markers',
            name='Predicted Return',
            line=dict(color='#4a90e2')
        ))
        
        fig2.add_trace(go.Scatter(
            x=results_df['test_end'],
            y=results_df['actual_return'],
            mode='lines+markers',
            name='Actual Return',
            line=dict(color='#e6f3ff')
        ))
        
        fig2.update_layout(
            title=f"{selected_model}: Predicted vs Actual Returns by Test Window",
            xaxis_title="Test End Date",
            yaxis_title="Return (%)",
            paper_bgcolor='#2f2f2f',
            plot_bgcolor='#2f2f2f',
            font=dict(color='#ffffff'),
            title_font=dict(color='#e6f3ff'),
            xaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff'),
            yaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff')
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Show detailed window results
        with st.expander("View Detailed Window Results", expanded=False):
            # Format results for display
            display_df = results_df.copy()
            
            # Format columns
            display_df['actual_return'] = display_df['actual_return'].map("{:.2f}%".format)
            display_df['predicted_return'] = display_df['predicted_return'].map("{:.2f}%".format)
            display_df['predicted_probability'] = display_df['predicted_probability'].map("{:.2f}%".format)
            display_df['price_error_pct'] = display_df['price_error_pct'].map("{:.2f}%".format)
            display_df['probability_error_pct'] = display_df['probability_error_pct'].map("{:.2f}%".format)
            
            # Select columns to display
            display_cols = [
                'test_start', 'test_end', 'start_price', 'final_price', 'predicted_final',
                'actual_return', 'predicted_return', 'direction_correct',
                'target_reached', 'predicted_probability'
            ]
            
            st.dataframe(display_df[display_cols], use_container_width=True)
        
        # Error analysis
        st.subheader("Error Analysis")
        
        # Calculate error distribution
        error_df = pd.DataFrame({
            'Price Error (%)': results_df['price_error_pct'],
            'Return Error (%)': abs(results_df['predicted_return'] - results_df['actual_return']),
            'Probability Error (%)': results_df['probability_error_pct']
        })
        
        # Bar chart for average errors
        fig3 = px.bar(
            x=['Price Error', 'Return Error', 'Probability Error'],
            y=[error_df['Price Error (%)'].mean(), 
               error_df['Return Error (%)'].mean(), 
               error_df['Probability Error (%)'].mean()],
            title=f"{selected_model}: Average Prediction Errors",
            labels={'x': 'Error Type', 'y': 'Error (%)'},
            color=['Price Error', 'Return Error', 'Probability Error'],
            color_discrete_map={
                'Price Error': '#4a90e2',
                'Return Error': '#6ab0ed',
                'Probability Error': '#8ab4f8'
            }
        )
        
        fig3.update_layout(
            paper_bgcolor='#2f2f2f',
            plot_bgcolor='#2f2f2f',
            font=dict(color='#ffffff'),
            title_font=dict(color='#e6f3ff'),
            xaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff'),
            yaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff')
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Correlation of predicted vs actual returns
        corr = np.corrcoef(results_df['predicted_return'], results_df['actual_return'])[0, 1]
        
        st.metric(
            label="Prediction-Actual Correlation",
            value=f"{corr:.3f}",
            delta=f"Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak",
            delta_color="normal" if corr > 0 else "inverse"
        )
    
    # Tab 3: Model Comparison
    with tab3:
        st.subheader("Model Comparison")
        
        # Get all models with results
        all_models = list(backtest_engine.model_results.keys())
        if backtest_engine.ensemble_results:
            all_models.append("Ensemble")
        
        if len(all_models) < 2:
            st.warning("Need at least two models to compare.")
            return
        
        # Create comparison data
        comparison_data = []
        
        for model in all_models:
            if model == "Ensemble":
                model_results = backtest_engine.ensemble_results
            else:
                model_results = backtest_engine.model_results.get(model, [])
            
            if not model_results:
                continue
                
            # Calculate metrics
            model_df = pd.DataFrame(model_results)
            
            comparison_data.append({
                'Model': model,
                'Direction Accuracy (%)': model_df['direction_correct'].mean() * 100,
                'Avg Price Error (%)': model_df['price_error_pct'].mean(),
                'Avg Return Error (%)': abs(model_df['predicted_return'] - model_df['actual_return']).mean(),
                'Return Bias (%)': model_df['predicted_return'].mean() - model_df['actual_return'].mean(),
                'Probability Error (%)': model_df['probability_error_pct'].mean()
            })
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create radar chart for model comparison
        categories = ['Direction Accuracy (%)', 'Avg Price Error (%)', 'Avg Return Error (%)', 
                     'Return Bias (%)', 'Probability Error (%)']
        
        # Scale factors (direction accuracy should be high, errors should be low)
        scale_factors = [1, -1, -1, -1, -1]  # 1 for metrics where higher is better, -1 where lower is better
        
        # Normalize data for radar chart (0-1 scale)
        radar_data = comparison_df.copy()
        
        for i, category in enumerate(categories):
            # Get max and min
            col_max = radar_data[category].max()
            col_min = radar_data[category].min()
            
            # Avoid division by zero
            if col_max == col_min:
                radar_data[category] = 0.5
            else:
                # Normalize (0-1)
                radar_data[category] = (radar_data[category] - col_min) / (col_max - col_min)
                
                # Apply scale factor
                if scale_factors[i] < 0:
                    radar_data[category] = 1 - radar_data[category]
        
        # Create radar chart
        fig4 = go.Figure()
        
        for i, model in enumerate(radar_data['Model']):
            fig4.add_trace(go.Scatterpolar(
                r=radar_data.loc[radar_data['Model'] == model, categories].values.flatten().tolist(),
                theta=categories,
                fill='toself',
                name=model
            ))
        
        fig4.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            paper_bgcolor='#2f2f2f',
            plot_bgcolor='#2f2f2f',
            font=dict(color='#ffffff'),
            title_font=dict(color='#e6f3ff'),
            title="Model Performance Comparison (higher is better)"
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
        # Display comparison table
        st.subheader("Raw Comparison Metrics")
        
        # Format metrics
        display_comparison = comparison_df.copy()
        display_comparison['Direction Accuracy (%)'] = display_comparison['Direction Accuracy (%)'].map("{:.1f}%".format)
        display_comparison['Avg Price Error (%)'] = display_comparison['Avg Price Error (%)'].map("{:.2f}%".format)
        display_comparison['Avg Return Error (%)'] = display_comparison['Avg Return Error (%)'].map("{:.2f}%".format)
        display_comparison['Return Bias (%)'] = display_comparison['Return Bias (%)'].map("{:.2f}%".format)
        display_comparison['Probability Error (%)'] = display_comparison['Probability Error (%)'].map("{:.2f}%".format)
        
        st.dataframe(display_comparison, use_container_width=True)
        
        # Recommendation
        st.subheader("Model Recommendations")
        
        # Calculate scores (weighted average of normalized metrics)
        model_scores = {}
        
        # Weights for different purposes
        weights = {
            'direction': [0.8, 0.1, 0.05, 0.0, 0.05],  # Focus on direction
            'price': [0.2, 0.5, 0.2, 0.05, 0.05],      # Focus on price accuracy
            'balanced': [0.3, 0.2, 0.2, 0.1, 0.2]      # Balanced approach
        }
        
        for purpose, weight in weights.items():
            for model in radar_data['Model'].unique():
                model_data = radar_data[radar_data['Model'] == model]
                if len(model_data) == 0:
                    continue
                    
                # Calculate weighted score
                score = sum(model_data[category].values[0] * w for category, w in zip(categories, weight))
                
                if model not in model_scores:
                    model_scores[model] = {}
                    
                model_scores[model][purpose] = score
        
        # Find best model for each purpose
        best_models = {}
        for purpose in weights.keys():
            purpose_scores = {model: scores[purpose] for model, scores in model_scores.items() if purpose in scores}
            if purpose_scores:
                best_model = max(purpose_scores, key=purpose_scores.get)
                best_score = purpose_scores[best_model]
                best_models[purpose] = (best_model, best_score)
        
        # Display recommendations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'direction' in best_models:
                model, score = best_models['direction']
                st.metric("Best for Direction", model, f"Score: {score:.2f}")
            else:
                st.metric("Best for Direction", "N/A", "Insufficient data")
        
        with col2:
            if 'price' in best_models:
                model, score = best_models['price']
                st.metric("Best for Price Accuracy", model, f"Score: {score:.2f}")
            else:
                st.metric("Best for Price Accuracy", "N/A", "Insufficient data")
        
        with col3:
            if 'balanced' in best_models:
                model, score = best_models['balanced']
                st.metric("Best Overall", model, f"Score: {score:.2f}")
            else:
                st.metric("Best Overall", "N/A", "Insufficient data")
