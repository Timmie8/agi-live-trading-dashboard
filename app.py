import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from ta_engine import StockAnalyzer

st.set_page_config(page_title="Technical Trader Command Center", layout="wide")

st.title("📈 Technical Trader Command Center")
st.caption("Gebaseerd op de trading regels: STO/MACD Crosses, RSI Overbought, Volume-bevestiging en Trendlijnen.")

# Sidebar Instellingen
st.sidebar.header("⚙️ Instellingen")
api_key = st.sidebar.text_input("Finnhub API Key", type="password")
symbol = st.sidebar.text_input("Aandeel Ticker", "SPY").upper()

timeframe = st.sidebar.selectbox("Time Frame (Frequency)", ["Dagelijks (D)", "30 minuten (30)", "15 minuten (15)", "5 minuten (5)"], index=0)
chart_type = st.sidebar.radio("Grafiek Type", ["Candlesticks", "Lijn"], index=0)
indicator_choice = st.sidebar.radio("Onderste Indicator", ["Slow-Stochastic (STO)", "RSI"], index=0)

tf_map = {"Dagelijks (D)": "D", "30 minuten (30)": "30", "15 minuten (15)": "15", "5 minuten (5)": "5"}

if not api_key:
    st.info("👈 Voer je Finnhub API Key in de sidebar in om te starten.")
    st.stop()

analyzer = StockAnalyzer(api_key=api_key)
df = analyzer.get_stock_data(symbol, resolution=tf_map[timeframe])

if df.empty:
    st.error(f"Geen data gevonden voor ticker '{symbol}'. Controleer je API-sleutel of ticker.")
    st.stop()

# Analyse resultaten opvragen
signals = analyzer.evaluate_signals(df)

# TOP METRICS DASHBOARD
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Laatste Prijs", f"${signals['Price']:.2f}", f"{signals['Change_Pct']}%")
m2.metric("RSI (14)", signals['RSI'], "OVERBOUGHT" if signals['RSI_Overbought'] else "Normaal")
m3.metric("Slow-STO", signals['STO_Status'])
m4.metric("MACD", signals['MACD_Status'])
m5.metric("Advies Signaal", signals['Action'])

# WAARSCHUWINGEN EN MOTIVATIE
with st.expander("📌 Trading Analyse & Bevestiging (Regels uit gids)", expanded=True):
    for r in signals['Reasons']:
        st.write(f"- {r}")

# FINANCIËLE GRAFIEK AANMAKEN (3 subplots: Prijs, Volume, Indicator)
fig = make_subplots(
    rows=3, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.03, 
    subplot_titles=(f"{symbol} Koers + Trendlijnen", "Volume", indicator_choice),
    row_width=[0.2, 0.2, 0.6]
)

# 1. Prijs Grafiek (Candlestick of Line)
if chart_type == "Candlesticks":
    fig.add_trace(go.Candlestick(
        x=df['Timestamp'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Candles"
    ), row=1, col=1)
else:
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Close'], mode='lines', name='Price', line=dict(color='blue')), row=1, col=1)

# Bepaal en teken Support & Resistance Niveaus (Automatisch de hoogste/laagste niveaus van de periode)
recent_high = df['High'].tail(30).max()
recent_low = df['Low'].tail(30).min()

fig.add_hline(y=recent_high, line_dash="dash", line_color="red", annotation_text="Resistance (Weerstand)", row=1, col=1)
fig.add_hline(y=recent_low, line_dash="dash", line_color="green", annotation_text="Support (Steun)", row=1, col=1)

# 2. Volume Grafiek
colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
fig.add_trace(go.Bar(x=df['Timestamp'], y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Vol_SMA20'], mode='lines', name='20 Volume Gemiddelde', line=dict(color='orange')), row=2, col=1)

# 3. Korte Termijn Indicatoren (RSI of Slow-STO)
if indicator_choice == "RSI":
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought (70)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold (30)", row=3, col=1)
else:
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Stoch_K'], mode='lines', name='Stoch %K', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Stoch_D'], mode='lines', name='Stoch %D (Signal)', line=dict(color='orange')), row=3, col=1)

fig.update_layout(height=750, xaxis_rangeslider_visible=False, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# SECTION: STOCK SCREENER (Breakout Scanner)
st.markdown("---")
st.header("🔍 Stock Screener (Breakout Scanner)")
st.write("Scant aandelen volgens je unieke parameters: Price Change > +5%, Hoog Volume en Prijs tussen $0.01 en $10.00.")

scan_list_input = st.text_input("In te scannen aandelen (gescheiden door komma's):", "AMD, F, SOFI, PLTR, NIO, SNAP, AAL, BAC")

if st.button("Start Screener"):
    tickers_to_scan = [t.strip().upper() for t in scan_list_input.split(",") if t.strip()]
    results = []

    for t in tickers_to_scan:
        scan_df = analyzer.get_stock_data(t, resolution="D", days_back=40)
        if not scan_df.empty and len(scan_df) >= 2:
            latest = scan_df.iloc[-1]
            prev = scan_df.iloc[-2]
            
            p_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
            vol = latest['Volume']
            avg_vol = latest['Vol_SMA20']
            price = latest['Close']

            # Jouw Screener Regels:
            # - Prijsverandering +5%
            # - Heavy volume (> 1.000.000 en boven gemiddelde)
            # - Prijs tussen 0.001 en 10 dollar
            is_price_change = p_change >= 5.0
            is_heavy_vol = vol > 1000000 and vol > avg_vol
            is_price_range = 0.001 <= price <= 10.0

            status = "🔥 POTENTIËLE BREAKOUT" if (is_price_change and is_heavy_vol and is_price_range) else "Geen Match"

            results.append({
                "Ticker": t,
                "Prijs ($)": round(price, 2),
                "Prijs Stijging (%)": round(p_change, 2),
                "Volume": int(vol),
                "Gem. Volume (20d)": int(avg_vol) if not pd.isna(avg_vol) else 0,
                "Binnen $0.01-$10 Range": "Ja" if is_price_range else "Nee",
                "Screener Match": status
            })

    res_df = pd.DataFrame(results)
    st.dataframe(res_df, use_container_width=True)
