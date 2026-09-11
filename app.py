import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from ta_engine import StockAnalyzer

st.set_page_config(page_title="Technical Trader + ML Command Center", layout="wide")

st.title("📈 Technical Trader Command Center + 🤖 ML Probability Engine")
st.caption("Combineert technische indicatoren (STO/MACD/RSI/Volume) met een Machine Learning Random Forest voorspellingsmodel.")

st.sidebar.header("⚙️ Instellingen")
symbol = st.sidebar.text_input("Aandeel Ticker", "SPY").upper()

timeframe_label = st.sidebar.selectbox("Time Frame (Frequency)", ["Dagelijks (1d)", "30 minuten (30m)", "15 minuten (15m)", "5 minuten (5m)"], index=0)
chart_type = st.sidebar.radio("Grafiek Type", ["Candlesticks", "Lijn"], index=0)
indicator_choice = st.sidebar.radio("Onderste Indicator", ["Slow-Stochastic (STO)", "RSI"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("🤖 ML Model Instellingen")
forecast_horizon = st.sidebar.slider("ML Voorspellingshorizon (candles vooruit)", 1, 10, 3, help="Aantal candles in de toekomst waarop het model traint.")

tf_map = {
    "Dagelijks (1d)": "1d", 
    "30 minuten (30m)": "30m", 
    "15 minuten (15m)": "15m", 
    "5 minuten (5m)": "5m"
}

analyzer = StockAnalyzer()
df = analyzer.get_stock_data(symbol, timeframe=tf_map[timeframe_label])

if df.empty:
    st.error(f"Geen data gevonden voor ticker '{symbol}'. Controleer of het symbool correct is.")
    st.stop()

signals = analyzer.evaluate_signals(df)
ml_res = analyzer.predict_ml_probability(df, forecast_horizon=forecast_horizon)

# TOP METRICS DASHBOARD INCLUSIEF ML PROBABILITY
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Laatste Prijs", f"${signals['Price']:.2f}", f"{signals['Change_Pct']}%")
m2.metric("RSI (14)", signals['RSI'], "OVERBOUGHT" if signals['RSI_Overbought'] else "Normaal")
m3.metric("Slow-STO", signals['STO_Status'])
m4.metric("MACD", signals['MACD_Status'])
m5.metric("Advies Signaal", signals['Action'])

# ML Metric Card
ml_color = "normal" if ml_res['up_prob'] >= 50 else "inverse"
m6.metric("🤖 ML Kans op Stijging", f"{ml_res['up_prob']}%", f"Horizon: +{forecast_horizon} candles")

# SECTIE: ML DETAILS & INDICATOR VERKLARING
col_ml, col_reasons = st.columns([1, 1])

with col_ml:
    with st.expander("🤖 Machine Learning Model Details", expanded=True):
        st.write(f"**Voorspelde kans op prijsstijging:** `{ml_res['up_prob']}%`")
        st.write(f"**Kans op prijsdaling/zijwaarts:** `{ml_res['down_prob']}%`")
        st.progress(ml_res['up_prob'] / 100)
        
        if 'feature_importances' in ml_res:
            st.markdown("**Belangrijkste indicatoren voor de ML-beslissing:**")
            feat_df = pd.DataFrame(list(ml_res['feature_importances'].items()), columns=['Indicator', 'Gewicht']).sort_values(by='Gewicht', ascending=False)
            st.dataframe(feat_df.head(4), use_container_width=True, hide_index=True)

with col_reasons:
    with st.expander("📌 Regels & Signalen Bevestiging", expanded=True):
        for r in signals['Reasons']:
            st.write(f"- {r}")

# FINANCIËLE GRAFIEK
fig = make_subplots(
    rows=3, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.03, 
    subplot_titles=(f"{symbol} Koers + Support/Resistance", "Volume", indicator_choice),
    row_width=[0.2, 0.2, 0.6]
)

if chart_type == "Candlesticks":
    fig.add_trace(go.Candlestick(
        x=df['Timestamp'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Candles"
    ), row=1, col=1)
else:
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Close'], mode='lines', name='Price', line=dict(color='blue')), row=1, col=1)

recent_high = df['High'].tail(30).max()
recent_low = df['Low'].tail(30).min()

fig.add_hline(y=recent_high, line_dash="dash", line_color="red", annotation_text="Resistance", row=1, col=1)
fig.add_hline(y=recent_low, line_dash="dash", line_color="green", annotation_text="Support", row=1, col=1)

colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
fig.add_trace(go.Bar(x=df['Timestamp'], y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
if 'Vol_SMA20' in df.columns:
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Vol_SMA20'], mode='lines', name='20 Vol Avg', line=dict(color='orange')), row=2, col=1)

if indicator_choice == "RSI":
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought (70)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold (30)", row=3, col=1)
else:
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Stoch_K'], mode='lines', name='Stoch %K', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Stoch_D'], mode='lines', name='Stoch %D', line=dict(color='orange')), row=3, col=1)

fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# SCREENER INCLUSIEF ML PROBABILITY
st.markdown("---")
st.header("🔍 Breakout Screener + ML Probability")
scan_list_input = st.text_input("In te scannen aandelen (gescheiden door komma's):", "AMD, F, SOFI, PLTR, NIO, SNAP, AAL, BAC")

if st.button("Start Screener"):
    tickers_to_scan = [t.strip().upper() for t in scan_list_input.split(",") if t.strip()]
    results = []

    for t in tickers_to_scan:
        scan_df = analyzer.get_stock_data(t, timeframe="1d")
        if not scan_df.empty and len(scan_df) >= 30:
            latest = scan_df.iloc[-1]
            prev = scan_df.iloc[-2]
            
            p_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
            vol = latest['Volume']
            avg_vol = latest['Vol_SMA20'] if 'Vol_SMA20' in scan_df.columns else vol
            price = latest['Close']

            # ML Probability ophalen
            ml_pred = analyzer.predict_ml_probability(scan_df, forecast_horizon=forecast_horizon)

            is_price_change = p_change >= 5.0
            is_heavy_vol = vol > 1000000 and vol > avg_vol
            is_price_range = 0.001 <= price <= 10.0

            status = "🔥 POTENTIËLE BREAKOUT" if (is_price_change and is_heavy_vol and is_price_range) else "Geen Match"

            results.append({
                "Ticker": t,
                "Prijs ($)": round(price, 2),
                "Stijging (%)": round(p_change, 2),
                "Volume": int(vol),
                "🤖 ML Kans op Stijging (%)": f"{ml_pred['up_prob']}%",
                "Breakout Match": status
            })

    res_df = pd.DataFrame(results)
    st.dataframe(res_df, use_container_width=True)
