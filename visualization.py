import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def display_volume_analysis(history):
    """Create and display volume analysis charts."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3],
                        subplot_titles=("Price", "Volume"))
    
    # Add price candlestick
    fig.add_trace(
        go.Candlestick(
            x=history.index,
            open=history['Open'],
            high=history['High'],
            low=history['Low'],
            close=history['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume bar chart
    colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in history.iterrows()]
    fig.add_trace(
        go.Bar(
            x=history.index,
            y=history['Volume'],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history['SMA_20'],
            line=dict(color='blue', width=1),
            name="20-day MA"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history['SMA_50'],
            line=dict(color='orange', width=1),
            name="50-day MA"
        ),
        row=1, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        title_text="Price and Volume Analysis",
        xaxis_rangeslider_visible=False
    )
    
    return fig

def display_technical_indicators(history_with_indicators):
    """Display technical indicators charts."""
    # Create a figure with secondary y-axis
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25],
                       subplot_titles=("Price and Bollinger Bands", "RSI", "MACD"))
    
    # Price and Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=history_with_indicators.index,
            y=history_with_indicators['Close'],
            line=dict(color='blue'),
            name="Price"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=history_with_indicators.index,
            y=history_with_indicators['BB_Upper'],
            line=dict(color='gray', width=1, dash='dash'),
            name="Upper BB"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=history_with_indicators.index,
            y=history_with_indicators['BB_Middle'],
            line=dict(color='gray', width=1),
            name="Middle BB"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=history_with_indicators.index,
            y=history_with_indicators['BB_Lower'],
            line=dict(color='gray', width=1, dash='dash'),
            name="Lower BB",
            fill='tonexty',
            fillcolor='rgba(200, 200, 200, 0.2)'
        ),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=history_with_indicators.index,
            y=history_with_indicators['RSI'],
            line=dict(color='purple'),
            name="RSI"
        ),
        row=2, col=1
    )
    
    # Add RSI overbought/oversold lines
    fig.add_trace(
        go.Scatter(
            x=[history_with_indicators.index[0], history_with_indicators.index[-1]],
            y=[70, 70],
            line=dict(color='red', width=1, dash='dash'),
            name="Overbought"
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[history_with_indicators.index[0], history_with_indicators.index[-1]],
            y=[30, 30],
            line=dict(color='green', width=1, dash='dash'),
            name="Oversold"
        ),
        row=2, col=1
    )
    
    # MACD
    fig.add_trace(
        go.Scatter(
            x=history_with_indicators.index,
            y=history_with_indicators['MACD'],
            line=dict(color='blue'),
            name="MACD"
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=history_with_indicators.index,
            y=history_with_indicators['MACD_Signal'],
            line=dict(color='red'),
            name="Signal"
        ),
        row=3, col=1
    )
    
    # MACD histogram
    colors = ['green' if val > 0 else 'red' for val in history_with_indicators['MACD_Hist']]
    fig.add_trace(
        go.Bar(
            x=history_with_indicators.index,
            y=history_with_indicators['MACD_Hist'],
            marker_color=colors,
            name="Histogram"
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Technical Indicators",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis ranges
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

def create_price_distribution_plot(ensemble, target_price):
    """Create a histogram plot of final price distributions"""
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    all_final_prices = []
    for model_type, result in ensemble.results.items():
        # Sample up to 1000 prices to avoid overcrowding the plot
        sample_size = min(1000, len(result['final_prices']))
        sampled_prices = np.random.choice(result['final_prices'], sample_size, replace=False)
        
        fig.add_trace(go.Histogram(
            x=sampled_prices,
            name=ensemble.models[model_type].name,
            opacity=0.6,
            nbinsx=50
        ))
        
        all_final_prices.extend(sampled_prices)
    
    # Add vertical lines
    fig.add_vline(
        x=target_price,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Target: ${target_price:.2f}"
    )
    
    fig.add_vline(
        x=ensemble.stock_data.price,
        line_width=2,
        line_color="black",
        annotation_text=f"Current: ${ensemble.stock_data.price:.2f}"
    )
    
    # Use st.session_state instead of ensemble.session_state
    fig.add_vline(
        x=st.session_state.ensemble_result['mean_price'],
        line_width=3,
        line_color="purple",
        annotation_text=f"Mean: ${st.session_state.ensemble_result['mean_price']:.2f}"
    )
    
    fig.update_layout(
        title=f"Final Price Distribution - All Models",
        xaxis_title="Price ($)",
        yaxis_title="Frequency",
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_model_comparison_plot(ensemble, target_price):
    """Create a bar chart comparing model probabilities"""
    model_names = [ensemble.models[mt].name for mt in ensemble.results.keys()]
    probabilities = [result['final_probability'] for result in ensemble.results.values()]
    
    # Sort by probability
    df = pd.DataFrame({
        'Model': model_names,
        'Probability (%)': probabilities
    }).sort_values(by='Probability (%)')
    
    fig = px.bar(
        df,
        y='Model',
        x='Probability (%)',
        orientation='h',
        title=f"Probability of Reaching ${target_price:.2f} by Model"
    )
    
    # Add ensemble probability line
    fig.add_vline(
        x=st.session_state.ensemble_result['final_probability'],
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Ensemble: {st.session_state.ensemble_result['final_probability']:.2f}%"
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return fig

def create_confidence_interval_plot(ensemble, target_price):
    """Create a plot showing confidence intervals for each model"""
    ci_low = []
    ci_high = []
    model_names = []
    
    for model_type, result in ensemble.results.items():
        if 'confidence_interval' in result:
            model_names.append(ensemble.models[model_type].name)
            ci_low.append(result['confidence_interval'][0])
            ci_high.append(result['confidence_interval'][1])
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Model': model_names,
        'Lower CI': ci_low,
        'Upper CI': ci_high
    }).sort_values(by='Lower CI')
    
    fig = go.Figure()
    
    # Add confidence intervals as horizontal bars
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Lower CI'], row['Upper CI']],
            y=[row['Model'], row['Model']],
            mode='lines',
            line=dict(width=8, color='royalblue'),
            name=row['Model']
        ))
        
        # Add markers for the bounds
        fig.add_trace(go.Scatter(
            x=[row['Lower CI']],
            y=[row['Model']],
            mode='markers',
            marker=dict(size=10, color='royalblue'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[row['Upper CI']],
            y=[row['Model']],
            mode='markers',
            marker=dict(size=10, color='royalblue'),
            showlegend=False
        ))
    
    # Add ensemble confidence interval
    ensemble_ci = st.session_state.ensemble_result['confidence_interval']
    fig.add_vline(x=ensemble_ci[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_vline(x=ensemble_ci[1], line_width=2, line_dash="dash", line_color="red")
    
    # Add annotation for ensemble CI
    fig.add_annotation(
        x=(ensemble_ci[0] + ensemble_ci[1])/2,
        y=len(df)-1,
        text=f"Ensemble 95% CI: [${ensemble_ci[0]:.2f}, ${ensemble_ci[1]:.2f}]",
        showarrow=False,
        yshift=20,
        font=dict(color="red")
    )
    
    # Add target price line
    fig.add_vline(
        x=target_price,
        line_width=2,
        line_color="green",
        annotation_text=f"Target Price: ${target_price:.2f}"
    )
    
    fig.update_layout(
        title="95% Confidence Intervals by Model",
        xaxis_title="Price ($)",
        showlegend=False
    )
    
    return fig