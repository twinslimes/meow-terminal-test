import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Import local modules
from data_utils import get_api_keys, fetch_additional_stock_data, calculate_technical_indicators
from visualization import create_price_distribution_plot, create_model_comparison_plot, create_confidence_interval_plot
from analysis import display_stock_analysis_section, display_technical_indicators_section, display_fundamental_analysis_section
from day_trader import display_day_trader_section
from backtesting import display_backtesting_section

# Import all model classes from models.py
from models import (
    ModelType, StockData, StockPriceModel, GeometricBrownianMotion, 
    AdvancedGBM, JumpDiffusionModel, HestonModel, GARCHModel,
    RegimeSwitchingModel, QuasiMonteCarloModel, VarianceGammaModel, 
    NeuralSDEModel, StockModelEnsemble, HAS_ARCH
)

# Suppress warnings in the UI
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Meow Terminal",
    page_icon="😺",
    layout="wide",
)

# Apply terminal-style CSS with black, blue, gray, and muted white-blue accents
terminal_css = """
<style>
    /* Main terminal theme */
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'VT323', monospace;
        color: #ffffff; /* White text for readability on dark backgrounds */
        caret-color: #4a90e2; /* Muted blue cursor for contrast */
    }
    
    /* Background and main container */
    .main {
        background-color: #2f2f2f; /* Dark gray background for uniformity */
        background-image: linear-gradient(rgba(74, 144, 226, 0.05) 50%, transparent 50%);
        background-size: 100% 4px;
    }
    
    /* Old Windows style border (muted white-blue) */
    div.stButton > button, .stSelectbox > div > div, div.stNumberInput > div > div {
        border: 2px solid #e6f3ff !important; /* Muted white-blue border */
        border-right: 2px solid #000 !important;
        border-bottom: 2px solid #000 !important;
        background-color: #4a90e2 !important; /* Muted blue for buttons */
        color: #ffffff !important; /* White text for contrast */
        font-family: 'VT323', monospace !important;
        font-size: 18px !important;
    }
    
    div.stButton > button:active {
        border: 2px solid #000 !important;
        border-right: 2px solid #e6f3ff !important;
        border-bottom: 2px solid #e6f3ff !important;
    }
    
    /* Text inputs */
    div.stTextInput > div > div > input {
        background-color: #2f2f2f; /* Dark gray background for inputs */
        color: #ffffff; /* White text */
        border: 1px solid #e6f3ff; /* Muted white-blue border */
        font-family: 'VT323', monospace !important;
        font-size: 18px !important;
    }
    
    /* Metrics */
    div.stMetric > div {
        background-color: #2f2f2f; /* Dark gray background for metrics */
        border: 1px solid #e6f3ff; /* Muted white-blue border */
        padding: 10px;
    }
    
    div.stMetric label {
        color: #e6f3ff !important; /* Muted white-blue labels */
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e6f3ff !important; /* Muted white-blue for headers */
        font-family: 'VT323', monospace !important;
    }
    
    /* Sidebar (match main background for uniformity) */
    section[data-testid="stSidebar"] {
        background-color: #2f2f2f; /* Dark gray, matching main background */
        border-right: 2px solid #e6f3ff; /* Muted white-blue border */
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        color: #e6f3ff !important; /* Muted white-blue text */
    }
    
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] button {
        font-family: 'VT323', monospace !important;
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        background-color: #2f2f2f !important; /* Dark gray background for charts */
    }
    
    /* Slider handle */
    .stSlider > div > div > div > div {
        background-color: #e6f3ff !important; /* Muted white-blue slider handle */
    }
    
    /* CRT effect overlay (muted blue-gray) */
    .main::before {
        content: " ";
        display: block;
        position: fixed;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(rgba(74, 144, 226, 0) 50%, rgba(47, 47, 47, 0.15) 50%), 
                    linear-gradient(90deg, rgba(74, 144, 226, 0.05), rgba(47, 47, 47, 0.02), rgba(74, 144, 226, 0.05));
        z-index: 999;
        background-size: 100% 2px, 3px 100%;
        pointer-events: none;
    }
    
    /* Terminal-style inputs and boxes */
    div.stTextInput, div.stNumberInput {
        background-color: #2f2f2f; /* Dark gray background for inputs */
    }
    
    /* Tables with terminal styling */
    div.stTable, div.dataframe {
        background-color: #2f2f2f !important; /* Dark gray background for tables */
        color: #ffffff !important; /* White text */
        border: 1px solid #e6f3ff !important; /* Muted white-blue border */
    }
    
    /* Tabs */
    button[role="tab"] {
        background-color: #2f2f2f !important; /* Dark gray background for tabs */
        color: #e6f3ff !important; /* Muted white-blue text */
        border: 1px solid #e6f3ff !important; /* Muted white-blue border */
    }
    
    button[role="tab"][aria-selected="true"] {
        background-color: #4a90e2 !important; /* Muted blue for selected tab */
        border-bottom: 2px solid #e6f3ff !important; /* Muted white-blue border */
    }
    
    div[role="tabpanel"] {
        background-color: #2f2f2f !important; /* Dark gray background for tab panels */
        border: 1px solid #e6f3ff !important; /* Muted white-blue border */
    }
    
    /* Success/info messages */
    div.stSuccessMessage, div.stInfoMessage {
        background-color: #4a90e2 !important; /* Muted blue for messages */
        color: #ffffff !important; /* White text */
    }
    
    /* Windows 95-style title bar for sections */
    .win95-header {
        background-color: #4a90e2; /* Muted blue for headers */
        color: #ffffff !important;
        font-weight: bold;
        padding: 2px 5px;
        font-family: 'VT323', monospace;
        border-top: 2px solid #e6f3ff;
        border-left: 2px solid #e6f3ff;
        border-right: 2px solid #000000;
        border-bottom: 2px solid #000000;
        margin-bottom: 5px;
    }
    
    /* Windows 95-style panel */
    .win95-panel {
        background-color: #2f2f2f; /* Dark gray for panels */
        border-top: 2px solid #e6f3ff;
        border-left: 2px solid #e6f3ff;
        border-right: 2px solid #000000;
        border-bottom: 2px solid #000000;
        padding: 5px;
        margin: 10px 0;
    }
    
    /* Expander styling */
    details {
        background-color: #2f2f2f !important; /* Dark gray background for expanders */
        border: 1px solid #e6f3ff !important; /* Muted white-blue border */
    }
    
    details summary {
        color: #e6f3ff !important; /* Muted white-blue text */
        font-family: 'VT323', monospace !important;
    }
    
    /* Special terminal blinking cursor (muted white-blue) */
    .terminal-cursor::after {
        content: "▌";
        animation: blink 1s step-end infinite;
        font-weight: bold;
        color: #e6f3ff; /* Muted white-blue cursor */
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    
    /* Fix for checkbox color */
    .stCheckbox label p {
        color: #e6f3ff !important; /* Muted white-blue for checkboxes */
    }
</style>
"""

# Function to create a Windows 95-style header
def win95_header(text):
    return f'<div class="win95-header">{text}</div>'

def display_homepage():
    """Display the homepage of the application"""
    st.markdown(terminal_css, unsafe_allow_html=True)
    
    # Title with terminal styling
    st.markdown("<h1 style='color: #e6f3ff; text-align: center;'>🐱 Meow Terminal 🐱</h1>", unsafe_allow_html=True)
    
    # Button to enter app
    if st.button("Enter Terminal", key="enter_app", use_container_width=True):
        st.session_state.show_dashboard = True
        st.rerun()
    
    # YouTube Video below button
    st.video("https://www.youtube.com/watch?v=l9QTwRn_vmc&t=1s&ab_channel=twinslimes")  # User's actual video
    
    # Footer
    st.markdown("""
    <div style="margin-top: 20px; text-align: center; color: #e6f3ff; font-size: 12px;">
        <p>© 2025 Meow Terminal</p>
    </div>
    """, unsafe_allow_html=True)

def clear_analysis_results():
    """Clear analysis results when stock data changes."""
    # Clear prediction results
    if 'ensemble' in st.session_state:
        del st.session_state.ensemble
    if 'ensemble_result' in st.session_state:
        del st.session_state.ensemble_result
    if 'target_price' in st.session_state:
        del st.session_state.target_price
    if 'T' in st.session_state:
        del st.session_state.T

def run_prediction_analysis(ticker, T, dt, M, target_price):
    """Run stock price prediction analysis with selected models"""
    if 'stock_data' not in st.session_state or st.session_state.stock_data is None:
        st.warning("Please fetch stock data first.")
        return
    
    stock_data = st.session_state.stock_data
    
    # Verify we're working with the correct ticker
    if stock_data.ticker != ticker:
        st.warning(f"Stock data is for {stock_data.ticker}, not {ticker}. Please fetch data for {ticker} first.")
        return
    
    with st.spinner("Running simulations - this may take a minute..."):
        # Create ensemble
        ensemble = StockModelEnsemble(stock_data)
        
        # Add selected models to ensemble
        if use_gbm:
            ensemble.add_model(GeometricBrownianMotion(stock_data))
        if use_advanced_gbm:
            ensemble.add_model(AdvancedGBM(stock_data))
        if use_qmc:
            ensemble.add_model(QuasiMonteCarloModel(stock_data))
        if use_jump:
            ensemble.add_model(JumpDiffusionModel(stock_data))
        if use_heston:
            ensemble.add_model(HestonModel(stock_data))
        if use_garch and HAS_ARCH:
            ensemble.add_model(GARCHModel(stock_data))
        if use_regime:
            ensemble.add_model(RegimeSwitchingModel(stock_data))
        if use_vg:
            ensemble.add_model(VarianceGammaModel(stock_data))
        if use_neural:
            ensemble.add_model(NeuralSDEModel(stock_data))
        
        # Calibrate all models
        ensemble.calibrate_all()
        
        # Run all simulations
        ensemble.run_all_simulations(T, dt, M, target_price)
        
        # Compute ensemble forecast
        ensemble_result = ensemble.compute_ensemble_forecast()
        
        # Store in session state
        st.session_state.ensemble = ensemble
        st.session_state.ensemble_result = ensemble_result
        st.session_state.target_price = target_price
        st.session_state.T = T
        # Save current ticker to track if ticker changes
        st.session_state.analysis_ticker = ticker
    
    st.success("Analysis completed!")
    display_prediction_results()

def display_prediction_results():
    """Display prediction results in main dashboard"""
    # Check if we have analysis results
    if 'ensemble' not in st.session_state or 'ensemble_result' not in st.session_state:
        st.info("No analysis results available. Please run a prediction analysis.")
        return
        
    ensemble = st.session_state.ensemble
    ensemble_result = st.session_state.ensemble_result
    target_price = st.session_state.target_price
    T = st.session_state.get('T', 1.0)  # Get time horizon with default of 1.0
    
    # Verify we're displaying the correct ticker
    current_ticker = st.session_state.get('current_ticker', '')
    analysis_ticker = st.session_state.get('analysis_ticker', '')
    
    if current_ticker != analysis_ticker:
        st.warning(f"The displayed analysis is for {analysis_ticker}, but the current selected ticker is {current_ticker}. Please run a new analysis.")
        if st.button("Clear Analysis Results"):
            clear_analysis_results()
            st.rerun()
        return
    
    st.markdown(win95_header("Probability Analysis Results"), unsafe_allow_html=True)
    
    col_prob1, col_prob2, col_prob3 = st.columns(3)
    with col_prob1:
        st.metric(
            label="Average Price Probability", 
            value=f"{ensemble_result['avg_probability']:.2f}%",
            help="Probability that the average price over the time period will be at or above target"
        )
    with col_prob2:
        st.metric(
            label="Final Price Probability", 
            value=f"{ensemble_result['final_probability']:.2f}%",
            help="Probability that the price at the end of the time period will be at or above target"
        )
    with col_prob3:
        st.metric(
            label="Maximum Price Probability", 
            value=f"{ensemble_result['max_probability']:.2f}%",
            help="Probability that the maximum price during the time period will be at or above target"
        )
    
    # Expected price and confidence interval
    st.markdown(win95_header("Price Projections"), unsafe_allow_html=True)
    ci_low, ci_high = ensemble_result['confidence_interval']
    st.metric(
        label="Expected Final Price", 
        value=f"${ensemble_result['mean_price']:.2f}",
        delta=f"95% CI: ${ci_low:.2f} to ${ci_high:.2f}"
    )
    
    # User-friendly dashboard first, then detailed tabs
    tab0, tab1, tab2, tab3 = st.tabs(["Dashboard", "Price Distribution", "Model Comparison", "Confidence Intervals"])
    
    with tab0:
        # Create the user-friendly dashboard
        st.markdown(win95_header(f"Investment Outlook for {ensemble.stock_data.ticker}"), unsafe_allow_html=True)
        
        # Determine sentiment
        final_prob = ensemble_result['final_probability']
        mean_price = ensemble_result['mean_price']
        current_price = ensemble.stock_data.price
        price_change_pct = (mean_price / current_price - 1) * 100
        
        # Create sentiment gauge and other dashboard elements
        display_dashboard_gauges(ensemble, ensemble_result, T, target_price, price_change_pct, final_prob, mean_price, current_price, ci_low, ci_high)
    
    with tab1:
        # Distribution of final prices
        fig1 = create_price_distribution_plot(ensemble, target_price)
        
        # Update figure for terminal theme
        fig1.update_layout(
            paper_bgcolor='#2f2f2f',
            plot_bgcolor='#2f2f2f',
            font=dict(color='#ffffff'),
            title_font=dict(color='#e6f3ff'),
            xaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff'),
            yaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff')
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
    with tab2:
        # Model comparison bar chart
        fig2 = create_model_comparison_plot(ensemble, target_price)
        
        # Update figure for terminal theme
        fig2.update_layout(
            paper_bgcolor='#2f2f2f',
            plot_bgcolor='#2f2f2f',
            font=dict(color='#ffffff'),
            title_font=dict(color='#e6f3ff'),
            xaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff'),
            yaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff')
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    with tab3:
        # Confidence intervals by model
        fig3 = create_confidence_interval_plot(ensemble, target_price)
        
        # Update figure for terminal theme
        fig3.update_layout(
            paper_bgcolor='#2f2f2f',
            plot_bgcolor='#2f2f2f',
            font=dict(color='#ffffff'),
            title_font=dict(color='#e6f3ff'),
            xaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff'),
            yaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff')
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    # Model weights
    st.markdown(win95_header("Model Weights in Ensemble"), unsafe_allow_html=True)
    model_weights_df = pd.DataFrame(
        ensemble_result['model_ranking'], 
        columns=["Model", "Weight (%)"]
    )
    st.dataframe(model_weights_df, use_container_width=True)

def display_dashboard_gauges(ensemble, ensemble_result, T, target_price, price_change_pct, final_prob, mean_price, current_price, ci_low, ci_high):
    """Display dashboard gauges and summary"""
    import plotly.graph_objects as go
    
    # Determine sentiment
    if final_prob > 75:
        sentiment = "Strongly Bullish"
        sentiment_color = "#4a90e2"
    elif final_prob > 55:
        sentiment = "Moderately Bullish"
        sentiment_color = "#6ab0ed"
    elif final_prob > 45:
        sentiment = "Neutral"
        sentiment_color = "#808080"
    elif final_prob > 25:
        sentiment = "Moderately Bearish"
        sentiment_color = "#8ab4f8"
    else:
        sentiment = "Strongly Bearish"
        sentiment_color = "#2c5282"
    
    # Display sentiment
    col_sent1, col_sent2 = st.columns([1, 2])
    
    with col_sent1:
        # Create a gauge chart for sentiment
        gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = final_prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Outlook: {sentiment}", 'font': {'color': '#e6f3ff'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#e6f3ff'},
                'bar': {'color': sentiment_color},
                'steps': [
                    {'range': [0, 25], 'color': '#2f2f2f'},
                    {'range': [25, 45], 'color': '#3f3f3f'},
                    {'range': [45, 55], 'color': '#4f4f4f'},
                    {'range': [55, 75], 'color': '#5f5f5f'},
                    {'range': [75, 100], 'color': '#4a90e2'}
                ],
                'threshold': {
                    'line': {'color': "#e6f3ff", 'width': 4},
                    'thickness': 0.75,
                    'value': final_prob
                }
            }
        ))
        
        gauge.update_layout(
            paper_bgcolor='#2f2f2f',
            font={'color': '#e6f3ff'},
            height=250
        )
        
        st.plotly_chart(gauge, use_container_width=True)
        
        # Key insights box
        st.markdown("""
        <div style="border: 1px solid #e6f3ff; padding: 10px; background-color: #2f2f2f;">
        <h3 style="color: #e6f3ff;">Key Insights</h3>
        <ul style="color: #ffffff;">
          <li><strong>Target Price:</strong> ${:.2f}</li>
          <li><strong>Current Price:</strong> ${:.2f}</li>
          <li><strong>Expected in {:.1f} years:</strong> ${:.2f} ({:.1f}%)</li>
          <li><strong>Probability of reaching target:</strong> {:.1f}%</li>
        </ul>
        </div>
        """.format(target_price, current_price, T, mean_price, price_change_pct, final_prob), 
        unsafe_allow_html=True)
    
    with col_sent2:
        # Create a simplified forecast chart
        all_final_prices = []
        for result in ensemble.results.values():
            all_final_prices.extend(result['final_prices'][:1000])  # Limit to 1000 per model
        
        # Create histogram with density curve
        fig = px.histogram(
            all_final_prices, 
            nbins=50,
            title=f"Price Forecast Distribution in {T:.1f} Years",
            opacity=0.7,
            histnorm='probability density',
            color_discrete_sequence=['#e6f3ff']
        )
        
        fig.add_vline(x=current_price, line_color='#ffffff', line_dash='solid', 
                      annotation_text="Current", annotation_position="top right", 
                      annotation_font=dict(color='#e6f3ff'))
                      
        fig.add_vline(x=target_price, line_color='#8ab4f8', line_dash='dash', 
                      annotation_text="Target", annotation_position="top right",
                      annotation_font=dict(color='#e6f3ff'))
                      
        fig.add_vline(x=mean_price, line_color='#6ab0ed', line_dash='solid', 
                      annotation_text="Expected", annotation_position="top right",
                      annotation_font=dict(color='#e6f3ff'))
        
        # Add confidence interval
        fig.add_vrect(
            x0=ci_low, x1=ci_high,
            fillcolor="#4a90e2", opacity=0.25,
            layer="below", line_width=0,
            annotation_text="95% Confidence Interval",
            annotation_position="bottom right",
            annotation_font=dict(color='#e6f3ff')
        )
        
        fig.update_layout(
            paper_bgcolor='#2f2f2f',
            plot_bgcolor='#2f2f2f',
            font=dict(color='#ffffff'),
            xaxis_title="Price ($)",
            yaxis_title="Probability Density",
            xaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff'),
            yaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff'),
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Simplified model ranking and insights
    st.markdown(win95_header("Model Insights"), unsafe_allow_html=True)
    col_rank1, col_rank2 = st.columns(2)
    
    with col_rank1:
        # Simple model weight ranking
        model_weights = pd.DataFrame(
            ensemble_result['model_ranking'][:5], 
            columns=["Model", "Weight (%)"]
        )
        
        st.markdown("<h3 style='color: #e6f3ff;'>Top 5 Models by Weight</h3>", unsafe_allow_html=True)
        st.dataframe(model_weights, use_container_width=True, hide_index=True)
        
        # Risk assessment
        risk_level = "High" if ensemble.stock_data.volatility > 0.3 else "Medium" if ensemble.stock_data.volatility > 0.15 else "Low"
        risk_color = "#2c5282" if risk_level == "High" else "#8ab4f8" if risk_level == "Medium" else "#4a90e2"
        
        st.markdown("<h3 style='color: #e6f3ff;'>Risk Assessment</h3>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:{risk_color};font-weight:bold;font-size:20px;'>{risk_level} Risk</span> (Volatility: {ensemble.stock_data.volatility:.2f})", unsafe_allow_html=True)
    
    with col_rank2:
        # Pass ensemble_result to the display_investment_summary function
        display_investment_summary(ensemble, ensemble_result, T, target_price, final_prob, mean_price, current_price, price_change_pct, ci_low, ci_high)

def display_investment_summary(ensemble, ensemble_result, T, target_price, final_prob, mean_price, current_price, price_change_pct, ci_low, ci_high):
    """Display investment summary and recommendation"""
    st.markdown("<h3 style='color: #e6f3ff;'>Investment Summary</h3>", unsafe_allow_html=True)
    
    # Generate a summary based on the analysis
    if price_change_pct > 20:
        growth_txt = "significant growth potential"
    elif price_change_pct > 10:
        growth_txt = "moderate growth potential"
    elif price_change_pct > 0:
        growth_txt = "slight growth potential"
    elif price_change_pct > -10:
        growth_txt = "slight downside risk"
    else:
        growth_txt = "significant downside risk"
    
    if final_prob > 70:
        prob_txt = "high probability"
    elif final_prob > 50:
        prob_txt = "moderate probability"
    elif final_prob > 30:
        prob_txt = "low probability"
    else:
        prob_txt = "very low probability"
    
    # Summary text
    summary = f"""
    <div style="border: 1px solid #e6f3ff; padding: 10px; background-color: #2f2f2f; font-family: 'VT323', monospace;">
    {ensemble.stock_data.ticker} shows {growth_txt} over the next {T:.1f} years, with a {prob_txt} ({final_prob:.1f}%) 
    of reaching the target price of ${target_price:.2f}. The expected price is ${mean_price:.2f}, 
    representing a {price_change_pct:.1f}% change from the current price of ${current_price:.2f}.
    <br><br>
    The 95% confidence interval ranges from ${ci_low:.2f} to ${ci_high:.2f}, indicating the range of 
    likely outcomes based on our multi-model ensemble approach.
    <br><br>
    This forecast is based on an ensemble of {len(ensemble.models)} advanced financial models, with 
    the most influential being {ensemble_result['model_ranking'][0][0]}.
    </div>
    """
    
    st.markdown(summary, unsafe_allow_html=True)
    
    # Investment recommendation
    if final_prob > 60 and price_change_pct > 15:
        recommendation = "Strong Buy"
        rec_color = "#4a90e2"
    elif final_prob > 50 and price_change_pct > 10:
        recommendation = "Buy"
        rec_color = "#6ab0ed"
    elif final_prob > 40 and price_change_pct > 0:
        recommendation = "Hold"
        rec_color = "#808080"
    elif final_prob > 30:
        recommendation = "Reduce"
        rec_color = "#8ab4f8"
    else:
        recommendation = "Sell"
        rec_color = "#2c5282"
    
    st.markdown("<h3 style='color: #e6f3ff;'>Recommendation</h3>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center; border: 1px solid #e6f3ff; padding: 5px; background-color: #2f2f2f;'><span style='color:{rec_color};font-weight:bold;font-size:24px;'>{recommendation}</span></div>", unsafe_allow_html=True)

def display_basic_dashboard(ticker):
    """Display basic dashboard with current price and chart"""
    if 'stock_data' in st.session_state and st.session_state.stock_data:
        stock_data = st.session_state.stock_data
        
        # Verify we're displaying data for the correct ticker
        if stock_data.ticker != ticker:
            st.warning(f"Displaying data for {stock_data.ticker}, but the current selected ticker is {ticker}. Please fetch data for {ticker}.")
            return
        
        # Display current price and basic metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Current Price", value=f"${stock_data.price:.2f}")
        
        with col2:
            st.metric(label="Annual Volatility", value=f"{stock_data.volatility:.2f}")
        
        with col3:
            # Calculate implied cost of equity
            implied_coe = stock_data.risk_free_rate + stock_data.volatility * 0.5
            st.metric(label="Implied Cost of Equity", value=f"{implied_coe*100:.2f}%")
        
        # Show historical price chart
        if stock_data.historical_data is not None:
            hist_data = stock_data.historical_data
            if 'Close' in hist_data.columns:
                fig = px.line(hist_data['Close'], title=f"{ticker} Historical Price")
                
                # Update figure for terminal theme
                fig.update_layout(
                    paper_bgcolor='#2f2f2f',
                    plot_bgcolor='#2f2f2f',
                    font=dict(color='#ffffff'),
                    title_font=dict(color='#e6f3ff'),
                    xaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff'),
                    yaxis=dict(gridcolor='#4a90e2', linecolor='#e6f3ff', zerolinecolor='#e6f3ff')
                )
                fig.update_traces(line_color='#e6f3ff')
                
                st.plotly_chart(fig, use_container_width=True)
                
        # Prediction model parameters
        st.markdown(win95_header("Price Prediction"), unsafe_allow_html=True)
        st.markdown("<p style='color: #ffffff;'>Use the controls below to set up and run a price prediction analysis.</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            T = st.slider("Time Horizon (years)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            st.session_state.T = T  # Store in session state for later use
        
        with col2:
            # Get current price from session
            current_price = stock_data.price
            
            # Target price input
            target_price = st.number_input(
                "Target Price ($)", 
                min_value=float(current_price * 0.5), 
                max_value=float(current_price * 2.0), 
                value=float(current_price * 1.2),
                step=0.01
            )
        
        # Calculate implied return
        implied_return = (target_price / current_price - 1) * 100
        annual_return = ((target_price / current_price) ** (1/T) - 1) * 100
        
        st.metric(
            label="Implied Total Return", 
            value=f"{implied_return:.2f}%",
            delta=f"{annual_return:.2f}% annually"
        )
        
        # Model selection
        st.markdown(win95_header("Select Models to Include"), unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            global use_gbm, use_advanced_gbm, use_qmc
            use_gbm = st.checkbox("Geometric Brownian Motion", value=True)
            use_advanced_gbm = st.checkbox("Advanced GBM", value=True)
            use_qmc = st.checkbox("Quasi Monte Carlo", value=True)
        
        with col2:
            global use_jump, use_heston, use_garch
            use_jump = st.checkbox("Jump Diffusion", value=True)
            use_heston = st.checkbox("Heston Stochastic Volatility", value=True)
            use_garch = st.checkbox("GARCH Volatility", value=HAS_ARCH)
        
        with col3:
            global use_regime, use_vg, use_neural
            use_regime = st.checkbox("Regime Switching", value=True)
            use_vg = st.checkbox("Variance Gamma", value=True)
            use_neural = st.checkbox("Neural SDE", value=True)
        
        # Advanced settings
        with st.expander("Advanced Settings", expanded=False):
            M = st.slider("Number of Simulations", min_value=1000, max_value=10000, value=5000, step=1000)
            dt = st.select_slider(
                "Time Step", 
                options=[1/252, 1/52, 1/26, 1/12], 
                value=1/12, 
                format_func=lambda x: f"{int(1/x)} times per year"
            )
        
        # Button to run simulations
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("Run Probability Analysis", key="run_analysis", use_container_width=True):
            run_prediction_analysis(ticker, T, dt, M, target_price)
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    """Main function to run the application"""
    # Apply terminal CSS
    st.markdown(terminal_css, unsafe_allow_html=True)
    
    # Initialize session state to track if dashboard should be shown
    if 'show_dashboard' not in st.session_state:
        st.session_state.show_dashboard = False
    
    # Check if we should show homepage or dashboard
    if not st.session_state.show_dashboard:
        display_homepage()
        return
    
    # Get API keys
    alpha_vantage_key, fred_api_key = get_api_keys()
    
    # Sidebar for navigation
    st.sidebar.markdown("<h1 style='color: #e6f3ff;'>Navigation</h1>", unsafe_allow_html=True)
    
    # All navigation options in a single dropdown
    selected_section = st.sidebar.selectbox(
        "Go to",
        ["Stock Dashboard", "Day Trader", "Backtesting", "Stock Analysis", "Technical Indicators", "Fundamental Analysis"]
    )
    
    # Return to homepage button
    if st.sidebar.button("Return to Homepage"):
        st.session_state.show_dashboard = False
        st.rerun()
    
    # Sidebar for inputs (common across sections)
    st.sidebar.markdown("<h2 style='color: #e6f3ff;'>Stock Selection</h2>", unsafe_allow_html=True)
    
    # Fake terminal prompt for stock ticker
    st.sidebar.markdown("<span style='color: #e6f3ff;'>C:\\STOCKS\\> Enter ticker:</span>", unsafe_allow_html=True)
    
    # User inputs for stock ticker
    ticker = st.sidebar.text_input("", value="AAPL", label_visibility="collapsed").upper()
    # Store current ticker in session state for verification
    st.session_state.current_ticker = ticker
    
    # Check if ticker has changed and clear analysis if needed
    if 'last_analyzed_ticker' in st.session_state and st.session_state.last_analyzed_ticker != ticker:
        clear_analysis_results()
    
    # Button to fetch data
    if st.sidebar.button("Fetch Stock Data", use_container_width=True):
        with st.spinner("Fetching data - Please wait..."):
            # Clear previous analysis results when fetching new data
            clear_analysis_results()
            
            # Initialize stock data for models
            stock_data = StockData(ticker, alpha_vantage_key, fred_api_key)
            stock_data.fetch_data()
            
            # Fetch additional data for analysis
            additional_data = fetch_additional_stock_data(ticker)
            
            # Calculate technical indicators
            if additional_data and 'history' in additional_data and not additional_data['history'].empty:
                technical_indicators = calculate_technical_indicators(additional_data['history'])
            else:
                technical_indicators = None
            
            # Store in session state for later use
            st.session_state.stock_data = stock_data
            st.session_state.additional_data = additional_data
            st.session_state.technical_indicators = technical_indicators
            st.session_state.last_analyzed_ticker = ticker
            
            st.sidebar.success(f"Data for {ticker} fetched successfully!")
            # Force refresh to reflect new data
            st.rerun()
    
    # Terminal breadcrumb path at top
    current_path = f"C:\\> STOCKS\\{ticker}\\{selected_section.upper().replace(' ', '_')}"
    st.markdown(f"<div style='color: #e6f3ff; font-family: monospace; margin-bottom: 10px;'>{current_path}</div>", unsafe_allow_html=True)
    
    # Display the appropriate content based on the selection
    if selected_section == "Stock Dashboard":
        # If we have analysis results for the current ticker, show them
        if ('ensemble_result' in st.session_state and 
            'analysis_ticker' in st.session_state and 
            st.session_state.analysis_ticker == ticker):
            display_prediction_results()
        else:
            # Otherwise show the basic dashboard
            display_basic_dashboard(ticker)
            
    elif selected_section == "Day Trader":
        display_day_trader_section(ticker)
        
    elif selected_section == "Backtesting":
        display_backtesting_section(ticker)
            
    elif selected_section == "Stock Analysis":
        display_stock_analysis_section(ticker)
        
    elif selected_section == "Technical Indicators":
        display_technical_indicators_section(ticker)
        
    elif selected_section == "Fundamental Analysis":
        display_fundamental_analysis_section(ticker)

# This is the entry point of the script
if __name__ == "__main__":
    main()
