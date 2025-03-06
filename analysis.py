import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Import local modules
from data_utils import format_value, generate_technical_signals
from visualization import display_volume_analysis, display_technical_indicators

def display_stock_analysis_section(ticker):
    """Display stock price and volume analysis."""
    st.header("Stock Price and Volume Analysis")
    
    if 'additional_data' not in st.session_state or st.session_state.additional_data is None:
        st.info(f"Please fetch data for {ticker} to view stock analysis.")
        return
    
    additional_data = st.session_state.additional_data
    
    if additional_data and 'history' in additional_data and not additional_data['history'].empty:
        # Display volume analysis
        volume_fig = display_volume_analysis(st.session_state.technical_indicators)
        st.plotly_chart(volume_fig, use_container_width=True)
        
        # Display trading statistics
        st.subheader("Trading Statistics")
        
        history = additional_data['history']
        last_month = history.iloc[-21:] if len(history) >= 21 else history
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_volume = history['Volume'].mean()
            recent_avg_volume = last_month['Volume'].mean()
            volume_change = (recent_avg_volume / avg_volume - 1) * 100
            
            st.metric(
                label="Avg. Daily Volume",
                value=format_value(avg_volume, "volume"),
                delta=f"{volume_change:.1f}%" if not np.isnan(volume_change) else None,
                delta_color="off"
            )
        
        with col2:
            daily_returns = history['Close'].pct_change().dropna()
            avg_daily_return = daily_returns.mean() * 100
            
            st.metric(
                label="Avg. Daily Return",
                value=f"{avg_daily_return:.2f}%",
                delta=f"{avg_daily_return*252:.1f}% annualized" if not np.isnan(avg_daily_return) else None,
                delta_color="off"
            )
        
        with col3:
            # Calculate average true range (ATR)
            high_low = history['High'] - history['Low']
            high_close = np.abs(history['High'] - history['Close'].shift())
            low_close = np.abs(history['Low'] - history['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            current_price = history['Close'].iloc[-1]
            atr_pct = (atr / current_price) * 100
            
            st.metric(
                label="ATR (14)",
                value=f"${atr:.2f}",
                delta=f"{atr_pct:.2f}% of price"
            )
        
        with col4:
            # Calculate realized volatility
            realized_vol = daily_returns.std() * np.sqrt(252) * 100
            
            st.metric(
                label="Realized Volatility",
                value=f"{realized_vol:.2f}%"
            )
        
        # More detailed statistics
        st.subheader("Price Action Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create a table of price statistics
            price_stats = {
                "Statistic": ["Current Price", "52-Week High", "52-Week Low", "200-Day MA", "50-Day MA", "20-Day MA"],
                "Value": [
                    f"${history['Close'].iloc[-1]:.2f}",
                    f"${history['High'].max():.2f}",
                    f"${history['Low'].min():.2f}",
                    f"${history['Close'].rolling(200).mean().iloc[-1]:.2f}" if len(history) >= 200 else "N/A",
                    f"${history['Close'].rolling(50).mean().iloc[-1]:.2f}" if len(history) >= 50 else "N/A",
                    f"${history['Close'].rolling(20).mean().iloc[-1]:.2f}" if len(history) >= 20 else "N/A"
                ]
            }
            
            price_stats_df = pd.DataFrame(price_stats)
            st.table(price_stats_df)
        
        with col2:
            # Create a table of return statistics
            return_stats = {
                "Period": ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
                "Return": [
                    f"{(history['Close'].iloc[-1] / history['Close'].iloc[-2] - 1) * 100:.2f}%" if len(history) >= 2 else "N/A",
                    f"{(history['Close'].iloc[-1] / history['Close'].iloc[-5] - 1) * 100:.2f}%" if len(history) >= 5 else "N/A",
                    f"{(history['Close'].iloc[-1] / history['Close'].iloc[-21] - 1) * 100:.2f}%" if len(history) >= 21 else "N/A",
                    f"{(history['Close'].iloc[-1] / history['Close'].iloc[-63] - 1) * 100:.2f}%" if len(history) >= 63 else "N/A",
                    f"{(history['Close'].iloc[-1] / history['Close'].iloc[-126] - 1) * 100:.2f}%" if len(history) >= 126 else "N/A",
                    f"{(history['Close'].iloc[-1] / history['Close'].iloc[-252] - 1) * 100:.2f}%" if len(history) >= 252 else "N/A"
                ]
            }
            
            return_stats_df = pd.DataFrame(return_stats)
            st.table(return_stats_df)
    else:
        st.error("No historical data available for analysis.")

def display_technical_indicators_section(ticker):
    """Display technical indicators and charts."""
    st.header("Technical Indicators")
    
    if 'technical_indicators' not in st.session_state or st.session_state.technical_indicators is None:
        st.info(f"Please fetch data for {ticker} to view technical indicators.")
        return
    
    technical_indicators = st.session_state.technical_indicators
    
    # Display technical indicators chart
    tech_fig = display_technical_indicators(technical_indicators)
    st.plotly_chart(tech_fig, use_container_width=True)
    
    # Display current technical indicator values
    st.subheader("Current Technical Indicator Values")
    
    # Get the latest values
    latest = technical_indicators.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
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
        st.markdown(f"<span style='color:{rsi_color};font-size:14px;'>{rsi_status}</span>", unsafe_allow_html=True)
    
    with col2:
        # MACD
        macd_value = latest['MACD']
        signal_value = latest['MACD_Signal']
        macd_diff = macd_value - signal_value
        macd_status = "Bullish" if macd_diff > 0 else "Bearish"
        macd_color = "green" if macd_diff > 0 else "red"
        
        st.metric(
            label="MACD",
            value=f"{macd_value:.2f}",
            delta=f"Signal: {signal_value:.2f}",
            delta_color="off"
        )
        st.markdown(f"<span style='color:{macd_color};font-size:14px;'>{macd_status} ({macd_diff:.2f})</span>", unsafe_allow_html=True)
    
    with col3:
        # Bollinger Bands
        price = latest['Close']
        bb_upper = latest['BB_Upper']
        bb_lower = latest['BB_Lower']
        bb_mid = latest['BB_Middle']
        
        # Determine position within bands
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
    
    with col4:
        # Moving Averages
        price = latest['Close']
        sma_20 = latest['SMA_20']
        sma_50 = latest['SMA_50']
        sma_200 = latest['SMA_200'] if 'SMA_200' in latest else None
        
        # Determine MA trend
        if sma_20 > sma_50 and price > sma_20:
            ma_status = "Strong Bullish"
            ma_color = "darkgreen"
        elif price > sma_20 and price > sma_50:
            ma_status = "Bullish"
            ma_color = "green"
        elif price < sma_20 and price < sma_50:
            ma_status = "Bearish"
            ma_color = "red"
        else:
            ma_status = "Neutral"
            ma_color = "gray"
        
        st.metric(
            label="Moving Averages",
            value=f"SMA20: ${sma_20:.2f}",
            delta=f"SMA50: ${sma_50:.2f}",
            delta_color="off"
        )
        st.markdown(f"<span style='color:{ma_color};font-size:14px;'>{ma_status}</span>", unsafe_allow_html=True)
    
    # Technical signals summary
    st.subheader("Technical Signals Summary")
    
    # Generate technical signals
    signals = generate_technical_signals(technical_indicators)
    
    # Display signals in expandable sections
    for category, category_signals in signals.items():
        with st.expander(f"{category} Signals", expanded=True if category == "Summary" else False):
            for signal in category_signals:
                signal_color = "green" if signal['direction'] == "bullish" else "red" if signal['direction'] == "bearish" else "gray"
                st.markdown(f"<span style='color:{signal_color};'>• {signal['description']}</span>", unsafe_allow_html=True)

def display_fundamentals(fundamentals, ticker):
    """Display fundamental analysis data."""
    st.subheader("Fundamental Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Market Cap", 
            value=format_value(fundamentals.get('market_cap'), "currency")
        )
        st.metric(
            label="P/E Ratio", 
            value=format_value(fundamentals.get('pe_ratio'), "ratio")
        )
        st.metric(
            label="Dividend Yield", 
            value=format_value(fundamentals.get('dividend_yield'), "percentage")
        )
    
    with col2:
        st.metric(
            label="EPS (TTM)", 
            value=format_value(fundamentals.get('eps'), "currency")
        )
        st.metric(
            label="Beta", 
            value=format_value(fundamentals.get('beta'), "number")
        )
        st.metric(
            label="Avg. Daily Volume", 
            value=format_value(fundamentals.get('avg_volume'), "volume")
        )
    
    with col3:
        fifty_two_high = fundamentals.get('fifty_two_week_high')
        fifty_two_low = fundamentals.get('fifty_two_week_low')
        current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
        
        # Calculate percentage from 52-week high and low
        pct_from_high = ((current_price / fifty_two_high) - 1) * 100 if fifty_two_high else None
        pct_from_low = ((current_price / fifty_two_low) - 1) * 100 if fifty_two_low else None
        
        st.metric(
            label="52-Week High", 
            value=format_value(fifty_two_high, "currency"),
            delta=f"{pct_from_high:.1f}%" if pct_from_high is not None else None
        )
        
        st.metric(
            label="52-Week Low", 
            value=format_value(fifty_two_low, "currency"),
            delta=f"{pct_from_low:.1f}%" if pct_from_low is not None else None
        )
        
        st.metric(
            label="Sector/Industry", 
            value=f"{fundamentals.get('sector', 'N/A')}"
        )

def display_fundamental_analysis_section(ticker):
    """Display fundamental analysis data."""
    st.header("Fundamental Analysis")
    
    if 'additional_data' not in st.session_state or st.session_state.additional_data is None:
        st.info(f"Please fetch data for {ticker} to view fundamental analysis.")
        return
    
    additional_data = st.session_state.additional_data
    
    if additional_data and 'fundamentals' in additional_data:
        # Display fundamental metrics
        display_fundamentals(additional_data['fundamentals'], ticker)
        
        # Additional fundamental analysis
        st.subheader("Valuation Metrics")
        
        # Get fundamental data
        fundamentals = additional_data['fundamentals']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Compute Price to Earnings ratio
            pe_ratio = fundamentals.get('pe_ratio')
            if pe_ratio:
                pe_status = "High" if pe_ratio > 25 else "Low" if pe_ratio < 15 else "Average"
                pe_color = "red" if pe_ratio > 25 else "green" if pe_ratio < 15 else "gray"
                
                st.metric(
                    label="P/E Ratio",
                    value=f"{pe_ratio:.2f}",
                    delta=pe_status,
                    delta_color="off"
                )
                st.markdown(f"<span style='color:{pe_color};font-size:14px;'>{pe_status} compared to market average</span>", unsafe_allow_html=True)
            else:
                st.metric(label="P/E Ratio", value="N/A")
        
        with col2:
            # Market Cap categorization
            market_cap = fundamentals.get('market_cap')
            if market_cap:
                if market_cap >= 200e9:
                    cap_category = "Mega Cap"
                elif market_cap >= 10e9:
                    cap_category = "Large Cap"
                elif market_cap >= 2e9:
                    cap_category = "Mid Cap"
                elif market_cap >= 300e6:
                    cap_category = "Small Cap"
                else:
                    cap_category = "Micro Cap"
                
                st.metric(
                    label="Market Cap",
                    value=format_value(market_cap, "currency"),
                    delta=cap_category,
                    delta_color="off"
                )
            else:
                st.metric(label="Market Cap", value="N/A")
        
        with col3:
            # Dividend information
            div_yield = fundamentals.get('dividend_yield')
            if div_yield:
                div_yield_pct = div_yield * 100
                div_status = "High" if div_yield_pct > 4 else "Low" if div_yield_pct < 1 else "Average"
                div_color = "green" if div_yield_pct > 4 else "gray" if div_yield_pct < 1 else "blue"
                
                st.metric(
                    label="Dividend Yield",
                    value=f"{div_yield_pct:.2f}%",
                    delta=div_status,
                    delta_color="off"
                )
                st.markdown(f"<span style='color:{div_color};font-size:14px;'>{div_status} yield compared to market average</span>", unsafe_allow_html=True)
            else:
                st.metric(label="Dividend Yield", value="N/A")
        
        # Risk metrics
        st.subheader("Risk Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Beta interpretation
            beta = fundamentals.get('beta')
            if beta:
                if beta > 1.5:
                    beta_desc = "High Volatility"
                    beta_color = "red"
                elif beta > 1:
                    beta_desc = "Above Market Volatility"
                    beta_color = "orange"
                elif beta > 0.8:
                    beta_desc = "Market-like Volatility"
                    beta_color = "blue"
                else:
                    beta_desc = "Low Volatility"
                    beta_color = "green"
                
                st.metric(
                    label="Beta",
                    value=f"{beta:.2f}",
                    delta=beta_desc,
                    delta_color="off"
                )
                st.markdown(f"<span style='color:{beta_color};font-size:14px;'>{beta_desc} relative to the market</span>", unsafe_allow_html=True)
            else:
                st.metric(label="Beta", value="N/A")
        
        with col2:
            # 52-week range position
            high = fundamentals.get('fifty_two_week_high')
            low = fundamentals.get('fifty_two_week_low')
            current = st.session_state.stock_data.price if st.session_state.stock_data else None
            
            if high and low and current:
                range_pct = (current - low) / (high - low) * 100
                range_desc = "Near High" if range_pct > 75 else "Near Low" if range_pct < 25 else "Mid Range"
                range_color = "green" if range_pct < 25 else "red" if range_pct > 75 else "gray"
                
                st.metric(
                    label="52-Week Range Position",
                    value=f"{range_pct:.1f}%",
                    delta=range_desc,
                    delta_color="off"
                )
                
                # Create a range indicator
                range_html = f"""
                <div style="margin-top:10px; width:100%; height:20px; background-color:#f0f0f0; border-radius:3px; position:relative;">
                    <div style="position:absolute; top:0; left:0; width:{range_pct}%; height:20px; background-color:{range_color}; border-radius:3px;"></div>
                    <div style="position:absolute; top:0; left:0; width:100%; height:20px; text-align:center; line-height:20px; color:black;">
                        {low:.2f} | {current:.2f} | {high:.2f}
                    </div>
                </div>
                """
                st.markdown(range_html, unsafe_allow_html=True)
            else:
                st.metric(label="52-Week Range Position", value="N/A")
        
        # Sector/Industry analysis
        st.subheader("Sector and Industry Analysis")
        sector = fundamentals.get('sector', 'N/A')
        industry = fundamentals.get('industry', 'N/A')
        
        st.markdown(f"**Sector:** {sector}")
        st.markdown(f"**Industry:** {industry}")
        
        if sector != 'N/A':
            st.markdown(f"Sector and industry analysis would typically compare {ticker}'s performance to sector averages and peers.")
            
            # Placeholder for sector comparison
            st.markdown("""
            The sector comparison would include:
            - Relative P/E ratio vs sector average
            - Revenue growth compared to sector
            - Profit margins vs industry standards
            - Market share analysis
            """)
            
            # Placeholder for additional fundamental analysis
            st.markdown("To implement a full sector comparison, we would need to fetch data for peer companies and sector indices.")
    else:
        st.error("No fundamental data available for analysis.")