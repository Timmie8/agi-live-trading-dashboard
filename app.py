import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta_engine import StockAnalyzer

st.set_page_config(page_title="AI Live Trading Dashboard", layout="wide")

analyzer = StockAnalyzer()

st.title("📈 AI Live Trading Dashboard")

# Sidebar
st.sidebar.header("⚙️ Instellingen")
ticker = st.sidebar.text_input("Aandeel Ticker", value="AAPL").upper()
timeframe = st.sidebar.selectbox("Timeframe", options=["1d", "15m", "5m"], index=0)
period = st.sidebar.selectbox("Historie Periode", options=["1y", "6mo", "1mo"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("🤖 ML & Tuning")
forecast_horizon = st.sidebar.slider("ML Voorspellingshorizon (candles)", 1, 10, 3)
enable_grid_search = st.sidebar.checkbox("Schakel GridSearchCV in", value=True)

if st.sidebar.button("Analyseer Aandeel") or ticker:
    df = analyzer.get_stock_data(symbol=ticker, timeframe=timeframe, period=period)

    if df.empty:
        st.error(f"Geen data gevonden voor ticker '{ticker}'. Controleer het symbool.")
    else:
        signals = analyzer.evaluate_signals(df)
        ml_res = analyzer.predict_ml_probability(
            df, 
            forecast_horizon=forecast_horizon, 
            use_grid_search=enable_grid_search
        )

        # 1. Bepaal status & kleur voor de RSI badge
        if signals['RSI_Overbought']:
            rsi_status_text = f"OVERBOUGHT ({signals['RSI']})"
            bg_color = "#FF4B4B"  # Rood
            text_color = "#FFFFFF"
        elif signals['RSI_Above_55']:
            if signals.get('RSI_Cross_55', False):
                rsi_status_text = f"🔥 BREAKOUT > 55 ({signals['RSI']})"
            else:
                rsi_status_text = f"BULLISH > 55 ({signals['RSI']})"
            bg_color = "#28A745"  # Groen
            text_color = "#FFFFFF"
        else:
            rsi_status_text = f"BEARISH < 55 ({signals['RSI']})"
            bg_color = "#6C757D"  # Grijs
            text_color = "#FFFFFF"

        # 2. Live Dashboard Balk
        st.markdown("### 📊 Live Dashboard & Signalen")

        m1, m2, m3, m4, m5, m6 = st.columns(6)

        with m1:
            st.metric("Laatste Prijs", f"${signals['Price']:.2f}", f"{signals['Change_Pct']}%")

        with m2:
            st.caption("RSI (14) Status")
            st.markdown(
                f"""
                <div style="
                    background-color: {bg_color};
                    color: {text_color};
                    padding: 8px 10px;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: bold;
                    font-size: 13px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    {rsi_status_text}
                </div>
                """,
                unsafe_allow_html=True
            )

        with m3:
            st.metric("Slow-STO", signals['STO_Status'])

        with m4:
            st.metric("MACD", signals['MACD_Status'])

        with m5:
            st.metric("Advies Signaal", signals['Action'])

        with m6:
            st.metric("🤖 ML Kans (+{f}d)".format(f=forecast_horizon), f"{ml_res['up_prob']}%")

        st.markdown("---")

        # 3. Technische Grafiek & Indicatoren
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            subplot_titles=('Koers & Candlesticks', 'Volume', 'RSI Indicator'),
            row_width=[0.2, 0.2, 0.6]
        )

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df['Timestamp'], open=df['Open'], high=df['High'], 
            low=df['Low'], close=df['Close'], name='Koers'
        ), row=1, col=1)

        # Volume
        fig.add_trace(go.Bar(
            x=df['Timestamp'], y=df['Volume'], name='Volume', 
            marker_color='lightblue'
        ), row=2, col=1)

        # RSI + Hulplijnen (30, 55, 70)
        fig.add_trace(go.Scatter(
            x=df['Timestamp'], y=df['RSI'], mode='lines', 
            name='RSI', line=dict(color='purple')
        ), row=3, col=1)

        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=55, line_dash="dash", line_color="green", annotation_text="Breakout (55)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

        fig.update_layout(height=750, showlegend=False, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 4. ML Details & Onderbouwing
        col_reasons, col_ml = st.columns(2)

        with col_reasons:
            with st.expander("📋 Signaal Onderbouwing", expanded=True):
                for r in signals['Reasons']:
                    st.write(r)

        with col_ml:
            with st.expander("🤖 Machine Learning Model Details", expanded=True):
                st.write(f"**Voorspelde kans op prijsstijging:** `{ml_res['up_prob']}%`")
                st.progress(ml_res['up_prob'] / 100)

                if enable_grid_search and "best_params" in ml_res:
                    st.markdown("**Gevonden Optimale Parameters (GridSearch):**")
                    st.json(ml_res["best_params"])

                if 'feature_importances' in ml_res:
                    st.markdown("**Top Gewichten Indicatoren:**")
                    feat_df = pd.DataFrame(
                        list(ml_res['feature_importances'].items()), 
                        columns=['Indicator', 'Gewicht']
                    ).sort_values(by='Gewicht', ascending=False)
                    st.dataframe(feat_df.head(4), use_container_width=True, hide_index=True)
