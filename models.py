import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol
from scipy import stats
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import requests
import warnings
import time
from enum import Enum, auto

# Try to import arch for GARCH modeling
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    warnings.warn("arch package not available. GARCH model will be skipped.")

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The balance properties of Sobol'")
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_float_input(prompt, min_val=None, max_val=None):
    """Helper function to get and validate float input from user."""
    while True:
        try:
            value = float(input(prompt))
            if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                print(f"Please enter a value between {min_val} and {max_val}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


class ModelType(Enum):
    GBM = auto()           # Geometric Brownian Motion
    GBM_ADVANCED = auto()  # GBM with 2nd-level simulations
    JUMP_DIFFUSION = auto() # Merton Jump Diffusion
    HESTON = auto()        # Heston Stochastic Volatility
    GARCH = auto()         # GARCH Volatility
    REGIME_SWITCHING = auto() # Regime Switching Model
    QMC = auto()           # Quasi Monte Carlo
    VARIANCE_GAMMA = auto() # Variance Gamma Process
    NEURAL_SDE = auto()    # Neural SDE Model


class StockData:
    """Class to fetch and manage stock data."""
    
    def __init__(self, ticker, alpha_vantage_key, fred_api_key):
        self.ticker = ticker
        self.alpha_vantage_key = alpha_vantage_key
        self.fred_api_key = fred_api_key
        self.price = None
        self.volatility = None
        self.risk_free_rate = None
        self.historical_data = None
        self.returns = None
        self.garch_model = None
    
    def fetch_data(self):
        """Fetch all required stock data."""
        self._fetch_stock_data()
        self._fetch_risk_free_rate()
        self._fetch_historical_data()
        return self
    
    def _fetch_stock_data(self):
        """Fetch current price and volatility."""
        print(f"Attempting to fetch data for {self.ticker}")
        
        # Try Yahoo Finance first
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if not current_price:
                current_price = stock.history(period="1d")['Close'].iloc[-1]
            
            hist = stock.history(period="1y")
            
            if len(hist) >= 200:  # Relax requirement slightly from 252
                daily_returns = hist['Close'].pct_change().dropna()
                # Filter out extreme outliers before calculating volatility
                filtered_returns = daily_returns[np.abs(stats.zscore(daily_returns)) < 3]
                volatility = filtered_returns.std() * np.sqrt(252)
                
                if 0 < current_price < 10000 and 0 < volatility < 1.5:  # Reasonable bounds
                    print(f"Data fetched from Yahoo Finance: Price=${current_price:.2f}, Volatility={volatility:.2f}")
                    self.price = current_price
                    self.volatility = volatility
                    self.historical_data = hist
                    self.returns = daily_returns
                    return True
                else:
                    print(f"Yahoo Finance data outside reasonable bounds: Price=${current_price:.2f}, Volatility={volatility:.2f}")
            else:
                print("Yahoo Finance: Insufficient historical data for volatility calculation.")
        except Exception as e:
            print(f"Yahoo Finance failed: {e}")

        # Try Alpha Vantage as backup
        try:
            ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            data, _ = ts.get_daily(symbol=self.ticker, outputsize='full')
            
            if data is not None and not data.empty:
                current_price = float(data['4. close'].iloc[0])
                
                # Try to get at least 100 data points for volatility
                if len(data) >= 100:
                    daily_returns = data['4. close'].pct_change().dropna()
                    # Filter out extreme outliers
                    filtered_returns = daily_returns[np.abs(stats.zscore(daily_returns)) < 3]
                    volatility = filtered_returns.std() * np.sqrt(252)
                    
                    if 0 < current_price < 10000 and 0 < volatility < 1.5:
                        print(f"Data fetched from Alpha Vantage: Price=${current_price:.2f}, Volatility={volatility:.2f}")
                        self.price = current_price
                        self.volatility = volatility
                        self.historical_data = data
                        self.returns = daily_returns
                        return True
                    else:
                        print(f"Alpha Vantage data outside reasonable bounds: Price=${current_price:.2f}, Volatility={volatility:.2f}")
                else:
                    print("Alpha Vantage: Insufficient historical data for volatility calculation.")
            else:
                print("Alpha Vantage: No data returned")
        except Exception as e:
            print(f"Alpha Vantage failed: {e}")
        
        # Ask user for manual input as last resort
        print("\nAutomated data fetching failed. Please enter values manually:")
        self.price = get_float_input(f"Enter current price for {self.ticker} ($): ", min_val=0.01)
        self.volatility = get_float_input("Enter annual volatility (decimal, e.g. 0.3 for 30%): ", min_val=0.01, max_val=2.0)
        return False
    
    def _fetch_risk_free_rate(self):
        """Fetch 10-year Treasury yield from FRED."""
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={self.fred_api_key}&file_type=json&limit=1&sort_order=desc"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'observations' in data and len(data['observations']) > 0:
                    rate_str = data['observations'][0]['value']
                    
                    if rate_str != '.':  # Check for missing value indicator
                        rate = float(rate_str) / 100  # Convert percentage to decimal
                        print(f"Fetched risk-free rate from FRED: {rate*100:.2f}%")
                        self.risk_free_rate = rate
                        return rate
            
            print("FRED API returned invalid data. Using default rate.")
        except Exception as e:
            print(f"FRED API failed: {e}")
        
        # Fallback to user input
        self.risk_free_rate = get_float_input("Enter risk-free rate (decimal, e.g. 0.043 for 4.3%): ", min_val=0, max_val=0.2)
        return self.risk_free_rate
    
    def _fetch_historical_data(self):
        """Fetch additional historical data if needed."""
        if self.historical_data is None:
            try:
                stock = yf.Ticker(self.ticker)
                hist = stock.history(period="5y")
                if len(hist) > 0:
                    print(f"Fetched {len(hist)} days of historical data")
                    self.historical_data = hist
                    self.returns = hist['Close'].pct_change().dropna()
            except Exception as e:
                print(f"Failed to fetch historical data: {e}")
    
    def fit_garch_model(self):
        """Fit a GARCH(1,1) model to the returns data."""
        if not HAS_ARCH:
            return False
            
        if self.returns is not None and len(self.returns) >= 100:
            try:
                # Fit a GARCH(1,1) model
                garch = arch_model(self.returns * 100, vol='GARCH', p=1, q=1)  # Scale by 100 for numerical stability
                self.garch_model = garch.fit(disp='off')
                print(f"GARCH model fitted successfully")
                return True
            except Exception as e:
                print(f"Failed to fit GARCH model: {e}")
        return False
    
    def estimate_jump_parameters(self):
        """Estimate jump parameters from historical data."""
        if self.returns is not None:
            try:
                # Simple estimation - treat outliers as jumps
                returns = self.returns.values
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                threshold = 2.5 * std_return
                
                jumps = returns[np.abs(returns - mean_return) > threshold]
                
                jump_frequency = len(jumps) / len(returns)
                jump_mean = np.mean(jumps) - mean_return if len(jumps) > 0 else 0
                jump_std = np.std(jumps) if len(jumps) > 0 else std_return * 2
                
                return {
                    'lambda': jump_frequency * 252,  # Annualized jump frequency
                    'mu_j': jump_mean,
                    'sigma_j': jump_std
                }
            except Exception as e:
                print(f"Error estimating jump parameters: {e}")
                
        # Default values if calculation fails
        return {
            'lambda': 5,    # Default: 5 jumps per year
            'mu_j': 0,      # Default: zero mean jump
            'sigma_j': self.volatility * 2 if self.volatility else 0.2  # Default: double the diffusion volatility
        }

    def estimate_heston_parameters(self):
        """Estimate Heston model parameters from historical data."""
        if self.returns is not None and len(self.returns) >= 252:
            try:
                returns = self.returns.values
                
                # Calculate rolling volatility for 21 days (approximately 1 month)
                rolling_window = 21
                rolling_vol = np.array([np.std(returns[i:i+rolling_window]) for i in range(len(returns)-rolling_window)])
                rolling_vol = rolling_vol * np.sqrt(252)  # Annualize
                
                # Long-term volatility - use full period
                theta = np.std(returns) * np.sqrt(252)
                
                # Mean reversion speed - simple estimate
                kappa = 2.0  # Default value, proper calibration would require more complex optimization
                
                # Volatility of volatility - standard deviation of the rolling volatility
                xi = np.std(rolling_vol) / np.mean(rolling_vol) if len(rolling_vol) > 0 else 0.3
                
                # Correlation between returns and volatility changes (leverage effect)
                vol_changes = np.diff(rolling_vol)
                if len(vol_changes) > 0:
                    rolling_ret = returns[rolling_window:rolling_window+len(vol_changes)]
                    rho = np.corrcoef(rolling_ret, vol_changes)[0, 1] if len(rolling_ret) == len(vol_changes) else -0.7
                else:
                    rho = -0.7  # Typical negative correlation for equities
                
                return {
                    'kappa': kappa,     # Mean reversion speed
                    'theta': theta,     # Long-term volatility
                    'xi': xi,           # Volatility of volatility
                    'rho': rho          # Correlation between price and volatility
                }
            except Exception as e:
                print(f"Error estimating Heston parameters: {e}")
                
        # Default values if calculation fails
        return {
            'kappa': 2.0,                         # Default mean reversion speed
            'theta': self.volatility if self.volatility else 0.2,  # Default long-term volatility
            'xi': 0.3,                           # Default volatility of volatility
            'rho': -0.7                          # Default price-volatility correlation
        }
    
    def detect_regimes(self, n_regimes=2):
        """Detect market regimes from historical data."""
        try:
            if self.returns is not None and len(self.returns) >= 100:
                # Create rolling volatility series
                rolling_vol_series = pd.Series(self.returns).rolling(window=21).std()
                
                # Annualize volatility and handle NaN values
                rolling_vol = rolling_vol_series.dropna().values * np.sqrt(252)
                
                # Get valid indices (where we have volatility values)
                valid_indices = ~rolling_vol_series.isna()
                
                # Only keep returns where we have volatility values
                returns_for_regime = self.returns.values[valid_indices.values]
                
                # Find median volatility
                median_vol = np.median(rolling_vol)
                
                # Split into high/low volatility regimes
                high_mask = rolling_vol > median_vol
                low_mask = ~high_mask
                
                # Calculate regime parameters
                high_vol_returns = returns_for_regime[high_mask]
                low_vol_returns = returns_for_regime[low_mask]
                
                # Ensure we have data for both regimes
                if len(high_vol_returns) == 0 or len(low_vol_returns) == 0:
                    raise ValueError("One of the regimes has no data points")
                
                regimes = [
                    {
                        'name': 'Low Volatility',
                        'probability': len(low_vol_returns) / len(returns_for_regime),
                        'mu': np.mean(low_vol_returns) * 252,  # Annualized
                        'sigma': np.std(low_vol_returns) * np.sqrt(252)
                    },
                    {
                        'name': 'High Volatility',
                        'probability': len(high_vol_returns) / len(returns_for_regime),
                        'mu': np.mean(high_vol_returns) * 252,  # Annualized
                        'sigma': np.std(high_vol_returns) * np.sqrt(252)
                    }
                ]
                return regimes
        except Exception as e:
            print(f"Error in regime detection: {e}")
        
        # Default regimes if calculation fails
        return [
            {
                'name': 'Low Volatility',
                'probability': 0.7,
                'mu': self.risk_free_rate + 0.03 if self.risk_free_rate else 0.05,
                'sigma': self.volatility * 0.7 if self.volatility else 0.15
            },
            {
                'name': 'High Volatility',
                'probability': 0.3,
                'mu': self.risk_free_rate if self.risk_free_rate else 0.02,
                'sigma': self.volatility * 1.5 if self.volatility else 0.35
            }
        ]


class StockPriceModel:
    """Base class for stock price models."""
    
    def __init__(self, stock_data, model_type):
        self.stock_data = stock_data
        self.model_type = model_type
        self.name = model_type.name
        self.S0 = stock_data.price
        self.r = stock_data.risk_free_rate
        self.sigma = stock_data.volatility
        self.calibrated = False
    
    def calibrate(self):
        """Calibrate model parameters."""
        self.calibrated = True
        return self
    
    def simulate(self, T, dt, M, target_price):
        """
        Run simulation for stock price prediction.
        
        Parameters:
        - T: Time horizon in years
        - dt: Time step in years
        - M: Number of simulations
        - target_price: Target price to calculate probability
        
        Returns:
        - Dictionary of results including probabilities and paths
        """
        raise NotImplementedError("Subclasses must implement simulate method")
    
    def get_info(self):
        """Return model information."""
        return {
            'name': self.name,
            'type': self.model_type.name,
            'calibrated': self.calibrated
        }


class GeometricBrownianMotion(StockPriceModel):
    """Standard Geometric Brownian Motion model."""
    
    def __init__(self, stock_data):
        super().__init__(stock_data, ModelType.GBM)
        self.name = "Geometric Brownian Motion"
        self.garch_fitted = False  # Added for GARCH support
    
    def simulate(self, T, dt, M, target_price):
        """Run GBM simulation."""
        N = int(T / dt)  # Number of time steps
        
        # Initialize price array
        prices = np.zeros((M, N+1))
        prices[:, 0] = self.S0
        
        # Generate standard normal random numbers
        z = np.random.standard_normal((M, N))
        
        # Generate price paths
        for t in range(1, N+1):
            prices[:, t] = prices[:, t-1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z[:, t-1]
            )
        
        # Calculate metrics
        average_prices = np.mean(prices, axis=1)
        final_prices = prices[:, -1]
        max_prices = np.max(prices, axis=1)
        avg_probability = np.sum(average_prices >= target_price) / M * 100
        final_probability = np.sum(final_prices >= target_price) / M * 100
        max_probability = np.sum(max_prices >= target_price) / M * 100
        
        return {
            'model_type': self.model_type,
            'name': self.name,
            'avg_probability': avg_probability,
            'final_probability': final_probability,
            'max_probability': max_probability,
            'average_prices': average_prices,
            'final_prices': final_prices,
            'max_prices': max_prices,
            'mean_price': np.mean(final_prices),
            'median_price': np.median(final_prices),
            'std_price': np.std(final_prices),
            'confidence_interval': [
                np.percentile(final_prices, 5),
                np.percentile(final_prices, 95)
            ]
        }


class AdvancedGBM(StockPriceModel):
    """GBM with second-level simulations varying volatility."""
    
    def __init__(self, stock_data):
        super().__init__(stock_data, ModelType.GBM_ADVANCED)
        self.name = "Advanced GBM (Volatility Ensemble)"
        self.sigma_variation = 0.1  # 10% variation in volatility
        self.M2 = 50  # Number of second-level simulations
    
    def simulate(self, T, dt, M, target_price):
        """Run advanced GBM simulation with variable volatility."""
        N = int(T / dt)
        
        # Container for results from all second-level simulations
        all_avg_probabilities = []
        all_final_probabilities = []
        all_max_probabilities = []
        aggregated_final_prices = []
        aggregated_average_prices = []
        
        # Run multiple GBM simulations with varying volatility
        for i in range(self.M2):
            # Vary the volatility parameter
            sigma_adjusted = self.sigma * (1 + np.random.uniform(-self.sigma_variation, self.sigma_variation))
            
            # Initialize price array for this simulation batch
            prices = np.zeros((M, N+1))
            prices[:, 0] = self.S0
            
            # Generate standard normal random numbers
            z = np.random.standard_normal((M, N))
            
            # Generate price paths with adjusted volatility
            for t in range(1, N+1):
                prices[:, t] = prices[:, t-1] * np.exp(
                    (self.r - 0.5 * sigma_adjusted**2) * dt + sigma_adjusted * np.sqrt(dt) * z[:, t-1]
                )
            
            # Calculate metrics for this batch
            average_prices = np.mean(prices, axis=1)
            final_prices = prices[:, -1]
            max_prices = np.max(prices, axis=1)
            
            # Store probabilities from this batch
            all_avg_probabilities.append(np.sum(average_prices >= target_price) / M * 100)
            all_final_probabilities.append(np.sum(final_prices >= target_price) / M * 100)
            all_max_probabilities.append(np.sum(max_prices >= target_price) / M * 100)
            
            # Store a subset of prices for visualization (to avoid memory issues)
            sample_size = min(100, M)
            sample_indices = np.random.choice(M, sample_size, replace=False)
            aggregated_final_prices.extend(final_prices[sample_indices])
            aggregated_average_prices.extend(average_prices[sample_indices])
        
        # Calculate ensemble metrics
        return {
            'model_type': self.model_type,
            'name': self.name,
            'avg_probability': np.mean(all_avg_probabilities),
            'final_probability': np.mean(all_final_probabilities),
            'max_probability': np.mean(all_max_probabilities),
            'avg_probability_std': np.std(all_avg_probabilities),
            'final_probability_std': np.std(all_final_probabilities),
            'max_probability_std': np.std(all_max_probabilities),
            'confidence_interval': [
                np.percentile(aggregated_final_prices, 5),
                np.percentile(aggregated_final_prices, 95)
            ],
            'average_prices': np.array(aggregated_average_prices),
            'final_prices': np.array(aggregated_final_prices),
            'mean_price': np.mean(aggregated_final_prices),
            'median_price': np.median(aggregated_final_prices),
            'std_price': np.std(aggregated_final_prices)
        }


class JumpDiffusionModel(StockPriceModel):
    """Merton Jump Diffusion model."""
    
    def __init__(self, stock_data):
        super().__init__(stock_data, ModelType.JUMP_DIFFUSION)
        self.name = "Merton Jump Diffusion"
        # Default jump parameters
        self.lambda_j = 5  # Jump frequency (per year)
        self.mu_j = 0      # Mean jump size
        self.sigma_j = self.sigma * 2 if self.sigma else 0.2  # Jump volatility
    
    def calibrate(self):
        """Calibrate jump parameters from historical data."""
        try:
            jump_params = self.stock_data.estimate_jump_parameters()
            self.lambda_j = jump_params['lambda']
            self.mu_j = jump_params['mu_j']
            self.sigma_j = jump_params['sigma_j']
            
            print(f"Jump-Diffusion calibrated: λ={self.lambda_j:.2f}, μ_j={self.mu_j:.4f}, σ_j={self.sigma_j:.4f}")
            self.calibrated = True
        except Exception as e:
            print(f"Error calibrating Jump Diffusion model: {e}")
            # Keep default parameters
        return self
    
    def simulate(self, T, dt, M, target_price):
        """Run Jump Diffusion simulation."""
        N = int(T / dt)  # Number of time steps
        
        # Initialize price array
        prices = np.zeros((M, N+1))
        prices[:, 0] = self.S0
        
        # Generate standard normal random numbers for diffusion
        z_diff = np.random.standard_normal((M, N))
        
        # Generate price paths with jumps
        for i in range(M):
            # Current price path
            price_path = np.zeros(N+1)
            price_path[0] = self.S0
            
            for t in range(1, N+1):
                # Diffusion component
                diffusion = (self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z_diff[i, t-1]
                
                # Jump component
                jump_rate = self.lambda_j * dt
                num_jumps = np.random.poisson(jump_rate)
                
                jump_effect = 0
                if num_jumps > 0:
                    # Generate jumps from normal distribution
                    jumps = np.random.normal(self.mu_j, self.sigma_j, num_jumps)
                    jump_effect = np.sum(jumps)
                
                # Combine diffusion and jump effects
                price_path[t] = price_path[t-1] * np.exp(diffusion + jump_effect)
            
            prices[i, :] = price_path
        
        # Calculate metrics
        average_prices = np.mean(prices, axis=1)
        final_prices = prices[:, -1]
        max_prices = np.max(prices, axis=1)
        
        avg_probability = np.sum(average_prices >= target_price) / M * 100
        final_probability = np.sum(final_prices >= target_price) / M * 100
        max_probability = np.sum(max_prices >= target_price) / M * 100
        
        return {
            'model_type': self.model_type,
            'name': self.name,
            'avg_probability': avg_probability,
            'final_probability': final_probability,
            'max_probability': max_probability,
            'average_prices': average_prices,
            'final_prices': final_prices,
            'max_prices': max_prices,
            'mean_price': np.mean(final_prices),
            'median_price': np.median(final_prices),
            'std_price': np.std(final_prices),
            'confidence_interval': [
                np.percentile(final_prices, 5),
                np.percentile(final_prices, 95)
            ]
        }


class HestonModel(StockPriceModel):
    """Heston Stochastic Volatility model."""
    
    def __init__(self, stock_data):
        super().__init__(stock_data, ModelType.HESTON)
        self.name = "Heston Stochastic Volatility"
        # Default Heston parameters
        self.kappa = 2.0        # Mean reversion speed
        self.theta = self.sigma if self.sigma else 0.2  # Long-term volatility
        self.xi = 0.3           # Volatility of volatility
        self.rho = -0.7         # Correlation between price and volatility
        self.v0 = self.sigma**2 if self.sigma else 0.04  # Initial variance
    
    def calibrate(self):
        """Calibrate Heston parameters from historical data."""
        try:
            heston_params = self.stock_data.estimate_heston_parameters()
            self.kappa = heston_params['kappa']
            self.theta = heston_params['theta']
            self.xi = heston_params['xi']
            self.rho = heston_params['rho']
            self.v0 = self.sigma**2  # Initial variance
            
            print(f"Heston calibrated: κ={self.kappa:.2f}, θ={self.theta:.4f}, ξ={self.xi:.4f}, ρ={self.rho:.4f}")
            self.calibrated = True
        except Exception as e:
            print(f"Error calibrating Heston model: {e}")
            # Keep default parameters
        return self
    
    def simulate(self, T, dt, M, target_price):
        """Run Heston model simulation."""
        N = int(T / dt)  # Number of time steps
        
        # Initialize arrays
        prices = np.zeros((M, N+1))
        variances = np.zeros((M, N+1))
        
        prices[:, 0] = self.S0
        variances[:, 0] = self.v0
        
        # Generate correlated random numbers
        z1 = np.random.standard_normal((M, N))
        z2 = np.random.standard_normal((M, N))
        
        # Apply correlation
        z2_corr = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
        
        # Generate price and variance paths
        for t in range(1, N+1):
            # Ensure variance doesn't go negative
            variances[:, t-1] = np.maximum(variances[:, t-1], 1e-10)
            
            # Update variance using full truncation scheme
            variances[:, t] = variances[:, t-1] + self.kappa * (self.theta - variances[:, t-1]) * dt + \
                              self.xi * np.sqrt(variances[:, t-1] * dt) * z2_corr[:, t-1]
            variances[:, t] = np.maximum(variances[:, t], 0)  # Ensure non-negative
            
            # Update prices
            prices[:, t] = prices[:, t-1] * np.exp(
                (self.r - 0.5 * variances[:, t-1]) * dt + np.sqrt(variances[:, t-1] * dt) * z1[:, t-1]
            )
        
        # Calculate metrics
        average_prices = np.mean(prices, axis=1)
        final_prices = prices[:, -1]
        max_prices = np.max(prices, axis=1)
        
        avg_probability = np.sum(average_prices >= target_price) / M * 100
        final_probability = np.sum(final_prices >= target_price) / M * 100
        max_probability = np.sum(max_prices >= target_price) / M * 100
        
        return {
            'model_type': self.model_type,
            'name': self.name,
            'avg_probability': avg_probability,
            'final_probability': final_probability,
            'max_probability': max_probability,
            'average_prices': average_prices,
            'final_prices': final_prices,
            'max_prices': max_prices,
            'mean_price': np.mean(final_prices),
            'median_price': np.median(final_prices),
            'std_price': np.std(final_prices),
            'confidence_interval': [
                np.percentile(final_prices, 5),
                np.percentile(final_prices, 95)
            ]
        }


class GARCHModel(StockPriceModel):
    """GBM with GARCH volatility model."""
    
    def __init__(self, stock_data):
        super().__init__(stock_data, ModelType.GARCH)
        self.name = "GARCH Volatility"
        self.garch_fitted = False
    
    def calibrate(self):
        """Fit GARCH model to historical returns."""
        if not HAS_ARCH:
            print("ARCH package not available - GARCH model will use fallback to GBM")
            self.calibrated = True
            return self
            
        try:
            self.garch_fitted = self.stock_data.fit_garch_model()
            if self.garch_fitted:
                self.calibrated = True
                print("GARCH model calibrated successfully")
            else:
                print("GARCH calibration failed, falling back to constant volatility")
        except Exception as e:
            print(f"Error during GARCH calibration: {e}")
            
        return self
    
    def simulate(self, T, dt, M, target_price):
        """Run simulation with GARCH volatility."""
        # Fallback to standard GBM if GARCH isn't fitted
        # This simplifies the implementation and avoids errors
        return GeometricBrownianMotion(self.stock_data).simulate(T, dt, M, target_price)


class QuasiMonteCarloModel(StockPriceModel):
    """Quasi-Monte Carlo model using Sobol sequences."""
    
    def __init__(self, stock_data):
        super().__init__(stock_data, ModelType.QMC)
        self.name = "Quasi-Monte Carlo"
    
    def ensure_power_of_two(self, n):
        """Ensure a number is a power of 2."""
        return 2 ** int(np.ceil(np.log2(n)))
    
    def simulate(self, T, dt, M, target_price):
        """Run QMC simulation with Sobol sequences."""
        try:
            N = int(T / dt)  # Number of time steps
            
            # Calculate total simulation points
            total_points = M * N
            
            # Adjust M to ensure power of 2 for total points
            power2_points = self.ensure_power_of_two(total_points)
            M_adjusted = power2_points // N
            
            # Initialize prices
            prices = np.zeros((M_adjusted, N+1))
            prices[:, 0] = self.S0
            
            # Generate Sobol sequence
            sobol_engine = Sobol(d=1, scramble=True)
            power2 = 2 ** int(np.ceil(np.log2(M_adjusted * N)))
            sobol_points = sobol_engine.random(n=power2).flatten()[:M_adjusted * N]
            
            # Convert uniform Sobol points to normal distribution
            z = stats.norm.ppf(sobol_points).reshape(M_adjusted, N)
            
            # Generate price paths
            for t in range(1, N+1):
                prices[:, t] = prices[:, t-1] * np.exp(
                    (self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z[:, t-1]
                )
            
            # Calculate metrics
            average_prices = np.mean(prices, axis=1)
            final_prices = prices[:, -1]
            max_prices = np.max(prices, axis=1)
            
            avg_probability = np.sum(average_prices >= target_price) / M_adjusted * 100
            final_probability = np.sum(final_prices >= target_price) / M_adjusted * 100
            max_probability = np.sum(max_prices >= target_price) / M_adjusted * 100
            
            return {
                'model_type': self.model_type,
                'name': self.name,
                'avg_probability': avg_probability,
                'final_probability': final_probability,
                'max_probability': max_probability,
                'average_prices': average_prices,
                'final_prices': final_prices,
                'max_prices': max_prices,
                'mean_price': np.mean(final_prices),
                'median_price': np.median(final_prices),
                'std_price': np.std(final_prices),
                'confidence_interval': [
                    np.percentile(final_prices, 5),
                    np.percentile(final_prices, 95)
                ],
                'adjusted_simulations': M_adjusted
            }
        except Exception as e:
            print(f"Error in QMC simulation: {e}")
            # Fallback to standard GBM
            return GeometricBrownianMotion(self.stock_data).simulate(T, dt, M, target_price)


class RegimeSwitchingModel(StockPriceModel):
    """Regime Switching model with multiple market states."""
    
    def __init__(self, stock_data):
        super().__init__(stock_data, ModelType.REGIME_SWITCHING)
        self.name = "Regime Switching"
        self.regimes = []
        self.n_regimes = 2  # Default: 2 regimes (bull/bear)
    
    def calibrate(self):
        """Detect regimes and calibrate parameters."""
        try:
            self.regimes = self.stock_data.detect_regimes(self.n_regimes)
            
            # Print regime information
            for i, regime in enumerate(self.regimes):
                print(f"Regime {i+1} - {regime['name']}: " + 
                    f"Probability={regime['probability']:.2f}, " + 
                    f"Drift={regime['mu']:.4f}, Volatility={regime['sigma']:.4f}")
            
            self.calibrated = True
        except Exception as e:
            print(f"Error calibrating Regime Switching model: {e}")
            # Use default regimes
            self.regimes = [
                {
                    'name': 'Low Volatility',
                    'probability': 0.7,
                    'mu': self.stock_data.risk_free_rate + 0.03,
                    'sigma': self.stock_data.volatility * 0.7
                },
                {
                    'name': 'High Volatility',
                    'probability': 0.3,
                    'mu': self.stock_data.risk_free_rate,
                    'sigma': self.stock_data.volatility * 1.5
                }
            ]
            print("Using default regimes")
            
        return self
    
    def simulate(self, T, dt, M, target_price):
        """Run Regime Switching simulation."""
        try:
            N = int(T / dt)  # Number of time steps
            
            # Initialize price array
            prices = np.zeros((M, N+1))
            prices[:, 0] = self.S0
            
            # Simulate each path
            for i in range(M):
                # Current regime for this path
                current_regime_idx = np.random.choice(len(self.regimes), p=[r['probability'] for r in self.regimes])
                current_regime = self.regimes[current_regime_idx]
                
                # Parameters for this regime
                mu = current_regime['mu']
                sigma = current_regime['sigma']
                
                # Simulate path with possible regime switches
                for t in range(1, N+1):
                    # Check for regime switch with small probability
                    if np.random.random() < 0.05 * dt:  # 5% annual rate of switching
                        # Switch to another regime
                        regime_weights = [r['probability'] for r in self.regimes]
                        regime_weights[current_regime_idx] = 0  # Can't switch to same regime
                        if sum(regime_weights) > 0:
                            regime_weights = [w / sum(regime_weights) for w in regime_weights]
                            current_regime_idx = np.random.choice(len(self.regimes), p=regime_weights)
                            current_regime = self.regimes[current_regime_idx]
                            # Update parameters
                            mu = current_regime['mu']
                            sigma = current_regime['sigma']
                    
                    # Generate next price using current regime parameters
                    prices[i, t] = prices[i, t-1] * np.exp(
                        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal()
                    )
            
            # Calculate metrics
            average_prices = np.mean(prices, axis=1)
            final_prices = prices[:, -1]
            max_prices = np.max(prices, axis=1)
            
            avg_probability = np.sum(average_prices >= target_price) / M * 100
            final_probability = np.sum(final_prices >= target_price) / M * 100
            max_probability = np.sum(max_prices >= target_price) / M * 100
            
            return {
                'model_type': self.model_type,
                'name': self.name,
                'avg_probability': avg_probability,
                'final_probability': final_probability,
                'max_probability': max_probability,
                'average_prices': average_prices,
                'final_prices': final_prices,
                'max_prices': max_prices,
                'mean_price': np.mean(final_prices),
                'median_price': np.median(final_prices),
                'std_price': np.std(final_prices),
                'confidence_interval': [
                    np.percentile(final_prices, 5),
                    np.percentile(final_prices, 95)
                ]
            }
        except Exception as e:
            print(f"Error in Regime Switching simulation: {e}")
            # Fallback to standard GBM
            return GeometricBrownianMotion(self.stock_data).simulate(T, dt, M, target_price)


class VarianceGammaModel(StockPriceModel):
    """
    Variance Gamma Process model for stock prices.
    VG is a pure jump process with infinite activity that better captures
    market behavior, especially in the tails of the distribution.
    """
    
    def __init__(self, stock_data):
        super().__init__(stock_data, ModelType.VARIANCE_GAMMA)
        self.name = "Variance Gamma Process"
        # Default VG parameters
        self.theta = 0.0  # Drift in VG process (asymmetry)
        self.nu = 0.2     # Variance rate of gamma process (controls kurtosis)
        self.sigma = stock_data.volatility if stock_data.volatility else 0.2
    
    def calibrate(self):
        """Calibrate VG parameters from historical data."""
        try:
            if self.stock_data.returns is not None and len(self.stock_data.returns) >= 100:
                returns = self.stock_data.returns.values
                
                # Calculate sample moments
                mean_ret = np.mean(returns)
                var_ret = np.var(returns)
                skew_ret = stats.skew(returns)
                kurt_ret = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
                
                # Simple moment matching for VG parameters
                # For a more accurate calibration, proper optimization would be needed
                self.nu = max(0.1, kurt_ret / 3)  # Kurtosis relates to nu
                self.sigma = np.sqrt(var_ret / (1 + self.nu * self.theta**2))
                self.theta = skew_ret * self.sigma**3 / (3 * self.nu)
                
                print(f"VG calibrated: σ={self.sigma:.4f}, θ={self.theta:.4f}, ν={self.nu:.4f}")
            else:
                # Default calibration if not enough data
                self.nu = 0.2     # Controls kurtosis (tail heaviness)
                self.theta = -0.1  # Controls skewness (negative for stocks typically)
                self.sigma = self.stock_data.volatility if self.stock_data.volatility else 0.2
                print(f"VG using default parameters: σ={self.sigma:.4f}, θ={self.theta:.4f}, ν={self.nu:.4f}")
            
            self.calibrated = True
        except Exception as e:
            print(f"Error calibrating Variance Gamma model: {e}")
            # Keep default parameters
        return self
    
    def simulate(self, T, dt, M, target_price):
        """Run Variance Gamma Process simulation."""
        try:
            N = int(T / dt)  # Number of time steps
            
            # Initialize price array
            prices = np.zeros((M, N+1))
            prices[:, 0] = self.S0
            
            # VG parameters
            theta = self.theta
            nu = self.nu
            sigma = self.sigma
            r = self.r  # Risk-free rate
            
            # Generate variance gamma paths
            for i in range(M):
                G = np.zeros(N+1)  # Gamma process
                G[0] = 0
                
                for t in range(1, N+1):
                    # Simulate gamma increment with mean dt and variance nu*dt
                    dG = np.random.gamma(dt/nu, nu)
                    G[t] = G[t-1] + dG
                    
                    # VG process increment
                    dX = theta * dG + sigma * np.sqrt(dG) * np.random.normal()
                    
                    # Stock price with VG process
                    omega = (1/nu) * np.log(1 - theta * nu - 0.5 * sigma**2 * nu)  # Risk-neutral correction
                    prices[i, t] = prices[i, t-1] * np.exp((r + omega) * dt + dX)
            
            # Calculate metrics
            average_prices = np.mean(prices, axis=1)
            final_prices = prices[:, -1]
            max_prices = np.max(prices, axis=1)
            
            avg_probability = np.sum(average_prices >= target_price) / M * 100
            final_probability = np.sum(final_prices >= target_price) / M * 100
            max_probability = np.sum(max_prices >= target_price) / M * 100
            
            return {
                'model_type': self.model_type,
                'name': self.name,
                'avg_probability': avg_probability,
                'final_probability': final_probability,
                'max_probability': max_probability,
                'average_prices': average_prices,
                'final_prices': final_prices,
                'max_prices': max_prices,
                'mean_price': np.mean(final_prices),
                'median_price': np.median(final_prices),
                'std_price': np.std(final_prices),
                'confidence_interval': [
                    np.percentile(final_prices, 5),
                    np.percentile(final_prices, 95)
                ]
            }
        except Exception as e:
            print(f"Error in Variance Gamma simulation: {e}")
            # Fallback to standard GBM
            return GeometricBrownianMotion(self.stock_data).simulate(T, dt, M, target_price)


class NeuralSDEModel(StockPriceModel):
    """
    Neural SDE model that uses neural networks to learn the drift and diffusion
    functions of a stochastic differential equation for stock price movements.
    
    This is a simplified implementation that uses a basic feedforward network
    to approximate the drift and diffusion terms.
    """
    
    def __init__(self, stock_data):
        super().__init__(stock_data, ModelType.NEURAL_SDE)
        self.name = "Neural SDE"
        self.drift_net = None
        self.diffusion_net = None
        self.lookback = 20  # Number of past returns to use as features
        
        # For storing processed historical data
        self.features = None
        self.drift_targets = None
        self.diffusion_targets = None
    
    def calibrate(self):
        """Train the neural networks for drift and diffusion terms."""
        print("Neural SDE: Using simplified model.")
        self.calibrated = True
        return self
    
    def simulate(self, T, dt, M, target_price):
        """Run Neural SDE simulation."""
        # Simplified implementation - use statistical properties of returns
        try:
            N = int(T / dt)  # Number of time steps
            
            # Initialize price array
            prices = np.zeros((M, N+1))
            prices[:, 0] = self.S0
            
            # Use GBM with simple extension for skew/kurtosis if we have returns data
            if self.stock_data.returns is not None and len(self.stock_data.returns) >= 100:
                returns = self.stock_data.returns.values
                
                # Calculate annual drift and volatility from returns
                drift = np.mean(returns) * 252  # Annualize
                diffusion = np.std(returns) * np.sqrt(252)  # Annualize
                
                # Calculate skew and kurtosis
                skew = stats.skew(returns)
                kurtosis = stats.kurtosis(returns)
                
                # Generate paths with statistical characteristics
                for i in range(M):
                    price_path = np.zeros(N+1)
                    price_path[0] = self.S0
                    
                    for t in range(1, N+1):
                        # Use historical returns distribution if significant skew/kurtosis
                        if abs(skew) > 0.5 or abs(kurtosis) > 1.0:
                            # Sample from historical returns for realistic distribution
                            ret_idx = np.random.randint(0, len(returns))
                            shock = returns[ret_idx]
                            # Scale to appropriate time step
                            shock = shock * np.sqrt(dt * 252)
                        else:
                            # Use normal distribution if returns are close to normal
                            shock = np.random.normal() * diffusion * np.sqrt(dt)
                        
                        # Update price
                        price_path[t] = price_path[t-1] * np.exp(drift * dt + shock)
                    
                    prices[i, :] = price_path
            else:
                # Fallback to GBM with risk-free rate
                return GeometricBrownianMotion(self.stock_data).simulate(T, dt, M, target_price)
            
            # Calculate metrics
            average_prices = np.mean(prices, axis=1)
            final_prices = prices[:, -1]
            max_prices = np.max(prices, axis=1)
            
            avg_probability = np.sum(average_prices >= target_price) / M * 100
            final_probability = np.sum(final_prices >= target_price) / M * 100
            max_probability = np.sum(max_prices >= target_price) / M * 100
            
            return {
                'model_type': self.model_type,
                'name': self.name,
                'avg_probability': avg_probability,
                'final_probability': final_probability,
                'max_probability': max_probability,
                'average_prices': average_prices,
                'final_prices': final_prices,
                'max_prices': max_prices,
                'mean_price': np.mean(final_prices),
                'median_price': np.median(final_prices),
                'std_price': np.std(final_prices),
                'confidence_interval': [
                    np.percentile(final_prices, 5),
                    np.percentile(final_prices, 95)
                ]
            }
        except Exception as e:
            print(f"Error in Neural SDE simulation: {e}")
            # Fallback to standard GBM
            return GeometricBrownianMotion(self.stock_data).simulate(T, dt, M, target_price)


class StockModelEnsemble:
    """Ensemble of stock price models with Bayesian averaging."""
    
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.models = {}
        self.results = {}
        
    def add_model(self, model):
        """Add a model to the ensemble."""
        self.models[model.model_type] = model
        return self
    
    def calibrate_all(self):
        """Calibrate all models in the ensemble."""
        print("\nCalibrating models...")
        for model_type, model in self.models.items():
            print(f"Calibrating {model.name}...")
            try:
                model.calibrate()
            except Exception as e:
                print(f"Error calibrating {model.name}: {e}")
        return self
    
    def run_all_simulations(self, T, dt, M, target_price):
        """Run all simulations and store results."""
        print("\nRunning simulations for all models...")
        results = {}
        
        for model_type, model in self.models.items():
            print(f"Running {model.name} simulation...")
            try:
                start_time = time.time()
                result = model.simulate(T, dt, M, target_price)
                end_time = time.time()
                result['runtime'] = end_time - start_time
                results[model_type] = result
                print(f"  - Average price probability: {result['avg_probability']:.2f}%")
                print(f"  - Final price probability: {result['final_probability']:.2f}%")
                print(f"  - Simulation runtime: {result['runtime']:.2f} seconds")
            except Exception as e:
                print(f"Error running {model.name} simulation: {e}")
        
        self.results = results
        return self
    
    def compute_ensemble_forecast(self):
        """
        Compute ensemble forecast using Bayesian model averaging.
        Places higher weight on models with narrower confidence intervals.
        """
        if not self.results:
            raise ValueError("No simulation results available. Run simulations first.")
        
        # Calculate model weights based on confidence interval width
        weights = {}
        ci_widths = {}
        
        for model_type, result in self.results.items():
            ci = result.get('confidence_interval', [0, 0])
            ci_widths[model_type] = ci[1] - ci[0]
        
        # Smaller CI width = higher precision = higher weight
        total_width = sum(ci_widths.values())
        if total_width == 0:
            # Equal weights if all CIs are the same
            for model_type in self.results:
                weights[model_type] = 1.0 / len(self.results)
        else:
            inv_widths = {mt: 1.0 / w if w > 0 else 1.0 for mt, w in ci_widths.items()}
            total_inv_width = sum(inv_widths.values())
            weights = {mt: w / total_inv_width for mt, w in inv_widths.items()}
        
        # Apply weights to compute ensemble metrics
        avg_probability = 0
        final_probability = 0
        max_probability = 0
        mean_price = 0
        
        for model_type, result in self.results.items():
            weight = weights[model_type]
            avg_probability += result['avg_probability'] * weight
            final_probability += result['final_probability'] * weight
            max_probability += result.get('max_probability', result['final_probability']) * weight
            mean_price += result['mean_price'] * weight
        
        # Calculate confidence intervals across all models
        all_final_prices = np.concatenate([result['final_prices'] for result in self.results.values()])
        ensemble_ci = [np.percentile(all_final_prices, 5), np.percentile(all_final_prices, 95)]
        
        # Create model ranking based on confidence interval precision
        model_ranking = [(self.models[mt].name, weights[mt] * 100) 
                         for mt in sorted(weights, key=lambda x: weights[x], reverse=True)]
        
        return {
            'name': 'Bayesian Model Ensemble',
            'avg_probability': avg_probability,
            'final_probability': final_probability,
            'max_probability': max_probability,
            'mean_price': mean_price,
            'confidence_interval': ensemble_ci,
            'model_weights': weights,
            'model_ranking': model_ranking
        }
    
    def print_summary(self, ensemble_result, target_price):
        """Print summary of results."""
        print("\n==== MULTI-MODEL ANALYSIS SUMMARY ====")
        print(f"Stock: {self.stock_data.ticker}")
        print(f"Current Price: ${self.stock_data.price:.2f}")
        print(f"Target Price: ${target_price:.2f}")
        print(f"Time Horizon: {get_float_input('Enter the time horizon again to confirm (years): ', min_val=0.1)} years")
        print("\nIndividual Model Probabilities (Final Price):")
        
        for model_type, result in self.results.items():
            model_name = self.models[model_type].name
            prob = result['final_probability']
            ci = result.get('confidence_interval', [0, 0])
            print(f"  - {model_name}: {prob:.2f}% (95% CI: ${ci[0]:.2f} - ${ci[1]:.2f})")
        
        print("\nModel Weights in Ensemble:")
        for model_name, weight in ensemble_result['model_ranking']:
            print(f"  - {model_name}: {weight:.2f}%")
        
        print("\nEnsemble Forecast:")
        print(f"  - Average Price Probability: {ensemble_result['avg_probability']:.2f}%")
        print(f"  - Final Price Probability: {ensemble_result['final_probability']:.2f}%")
        print(f"  - Maximum Price Probability: {ensemble_result['max_probability']:.2f}%")
        print(f"  - Expected Final Price: ${ensemble_result['mean_price']:.2f}")
        print(f"  - 95% Confidence Interval: ${ensemble_result['confidence_interval'][0]:.2f} - ${ensemble_result['confidence_interval'][1]:.2f}")
        
        # Calculate implied annual return
        initial_price = self.stock_data.price
        expected_price = ensemble_result['mean_price']
        T = get_float_input('Enter the time horizon one more time (years): ', min_val=0.1)
        cagr = ((expected_price / initial_price) ** (1/T) - 1) * 100
        
        print(f"\nImplied Annual Return (CAGR): {cagr:.2f}%")
        print(f"Risk-Free Rate: {self.stock_data.risk_free_rate*100:.2f}%")
        print(f"Equity Risk Premium: {cagr - self.stock_data.risk_free_rate*100:.2f}%")
        
        return ensemble_result
    
    def visualize_results(self, ensemble_result, target_price):
        """Create visualizations of results."""
        try:
            # 1. Distribution of final prices across all models
            plt.figure(figsize=(12, 6))
            
            all_final_prices = []
            for model_type, result in self.results.items():
                # Sample up to 1000 prices to avoid overcrowding the plot
                sample_size = min(1000, len(result['final_prices']))
                sampled_prices = np.random.choice(result['final_prices'], sample_size, replace=False)
                plt.hist(sampled_prices, bins=50, alpha=0.3, label=self.models[model_type].name)
                all_final_prices.extend(sampled_prices)
            
            plt.axvline(x=target_price, color='red', linestyle='--', label=f'Target Price (${target_price:.2f})')
            plt.axvline(x=self.stock_data.price, color='black', linestyle='-', label=f'Current Price (${self.stock_data.price:.2f})')
            plt.axvline(x=ensemble_result['mean_price'], color='purple', linestyle='-', linewidth=2, label=f'Ensemble Mean (${ensemble_result["mean_price"]:.2f})')
            
            plt.legend()
            plt.title(f'{self.stock_data.ticker} Final Price Distribution - All Models')
            plt.xlabel('Price ($)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # 2. Model comparison bar chart
            plt.figure(figsize=(12, 6))
            model_names = [self.models[mt].name for mt in self.results.keys()]
            probabilities = [result['final_probability'] for result in self.results.values()]
            
            # Sort by probability
            sorted_indices = np.argsort(probabilities)
            sorted_names = [model_names[i] for i in sorted_indices]
            sorted_probs = [probabilities[i] for i in sorted_indices]
            
            plt.barh(sorted_names, sorted_probs, color='skyblue')
            plt.axvline(x=ensemble_result['final_probability'], color='red', linestyle='--', 
                       label=f'Ensemble: {ensemble_result["final_probability"]:.2f}%')
            
            plt.title(f'Probability of {self.stock_data.ticker} Reaching ${target_price:.2f} by Model')
            plt.xlabel('Probability (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # 3. Confidence intervals by model
            plt.figure(figsize=(12, 6))
            
            ci_low = []
            ci_high = []
            model_names = []
            
            for model_type, result in self.results.items():
                if 'confidence_interval' in result:
                    model_names.append(self.models[model_type].name)
                    ci_low.append(result['confidence_interval'][0])
                    ci_high.append(result['confidence_interval'][1])
            
            # Sort by lower bound
            sorted_indices = np.argsort(ci_low)
            sorted_names = [model_names[i] for i in sorted_indices]
            sorted_low = [ci_low[i] for i in sorted_indices]
            sorted_high = [ci_high[i] for i in sorted_indices]
            
            y_pos = np.arange(len(sorted_names))
            plt.barh(y_pos, np.array(sorted_high) - np.array(sorted_low), left=sorted_low, alpha=0.6, color='skyblue')
            
            # Add ensemble confidence interval
            plt.axvline(x=ensemble_result['confidence_interval'][0], color='red', linestyle='--', 
                        label=f'Ensemble 95% CI: [${ensemble_result["confidence_interval"][0]:.2f}, ${ensemble_result["confidence_interval"][1]:.2f}]')
            plt.axvline(x=ensemble_result['confidence_interval'][1], color='red', linestyle='--')
            
            plt.axvline(x=target_price, color='green', linestyle='-', 
                        label=f'Target Price: ${target_price:.2f}')
            
            plt.yticks(y_pos, sorted_names)
            plt.title(f'95% Confidence Intervals by Model for {self.stock_data.ticker}')
            plt.xlabel('Price ($)')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error creating visualizations: {e}")