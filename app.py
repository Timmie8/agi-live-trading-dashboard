import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import time
from agi_trading_engine import Senses, Subconscious, Mind

st.set_page_config(page_title="AGI Live Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("🧠 AGI Sentient-Architecture Trading Dashboard")
st.caption("Based on 'pftq' (2015) AGI Framework: Mind, Subconscious (Simulator), and Senses integrated with Finnhub Live API.")

# Sidebar Controls
st.sidebar.header("🔌 Configuration")
api_key = st.sidebar.text_input("Finnhub API Key", type="password", help="Enter your Finnhub.io API key")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", "AAPL, NVDA, TSLA, MSFT")
refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 3, 60, 5)

st.sidebar.subheader("🧠 Mind Internal Parameters")
risk_appetite = st.sidebar.slider("Internal Risk Appetite (Dynamic Objective)", 0.0, 1.0, 0.5, step=0.05)
sim_count = st.sidebar.slider("Subconscious Simulation Count", 20, 500, 100)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if not api_key:
    st.info("👈 Please enter your **Finnhub API Key** in the sidebar to start live analysis.")
    st.stop()

# Initialize Architecture Components
senses = Senses(api_key=api_key)
subconscious = Subconscious(num_simulations=sim_count)
mind = Mind(risk_appetite=risk_appetite)

# Main App Layout
selected_ticker = st.selectbox("Select Asset for Deep Subconscious Inspection", tickers)

col_left, col_right = st.columns([1, 1])

# Run Engine Execution
market_data = senses.fetch_market_state(selected_ticker)

if "error" in market_data and market_data["price"] == 0:
    st.error(f"Error fetching data for {selected_ticker}: {market_data['error']}")
else:
    sim_data = subconscious.run_mental_simulation(market_data["price"])
    decision = mind.evaluate(market_data, sim_data)
    causal_graph = subconscious.build_causal_knowledge_graph(market_data, sim_data)

    # LEFT COLUMN: SENSES & MIND DECISION
    with col_left:
        st.subheader("👁️ Senses (Perception) & 🧠 Mind (Decision)")
        
        # Metric Cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Live Price", f"${market_data['price']:.2f}", f"{market_data['change_pct']:.2f}%")
        m2.metric("Mind Action", decision["action"])
        m3.metric("Conviction", decision["conviction"], f"{decision['confidence']}% Conf.")

        st.markdown(f"**Reasoning Chain:** {decision['reasoning']}")

        # Display Ticker Overview Table
        st.markdown("### 📊 Portfolio Scan")
        scan_results = []
        for t in tickers:
            m_data = senses.fetch_market_state(t)
            s_data = subconscious.run_mental_simulation(m_data["price"])
            dec = mind.evaluate(m_data, s_data)
            scan_results.append({
                "Ticker": t,
                "Price ($)": m_data.get("price", 0.0),
                "Change (%)": round(m_data.get("change_pct", 0.0), 2),
                "Simulated Target": round(s_data.get("mean_projected", 0.0), 2),
                "Bull Odds (%)": round(s_data.get("bull_ratio", 0.0) * 100, 1),
                "Action": dec["action"]
            })
        st.dataframe(pd.DataFrame(scan_results), use_container_width=True)

    # RIGHT COLUMN: SUBCONSCIOUS SIMULATION & KNOWLEDGE GRAPH
    with col_right:
        st.subheader("🔮 Subconscious (What-If World Simulator)")
        
        # Plot Monte Carlo Simulations from Subconscious
        fig_sim = go.Figure()
        trajectories = sim_data.get("trajectories", [])
        for step in trajectories[:30]:  # Plot first 30 for UI responsiveness
            fig_sim.add_trace(go.Scatter(y=step, mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
        
        fig_sim.update_layout(
            title=f"Subconscious Trajectory Simulations for {selected_ticker}",
            xaxis_title="Forward Steps (Internal Simulation Time)",
            yaxis_title="Projected Price ($)",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        # Plot Knowledge Graph Schema
        st.markdown("### 🕸️ Causal Knowledge Graph (Profiles)")
        nodes = list(causal_graph.nodes())
        edges = list(causal_graph.edges())
        
        st.json({
            "Entities": nodes,
            "Causal Links": [{"From": u, "To": v, "Relation": causal_graph.edges[u, v]['relationship']} for u, v in edges]
        })

    # Auto-refresh mechanism
    time.sleep(refresh_rate)
    st.rerun()
