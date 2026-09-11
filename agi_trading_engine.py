import time
import finnhub
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime

class Senses:
    """
    PERCEPTION LAYER:
    Extracts raw data from Finnhub, normalizes it, and abstracts high-dimensional
    market data into structural signal primitives (Profiles).
    """
    def __init__(self, api_key: str):
        self.client = finnhub.Client(api_key=api_key)

    def fetch_market_state(self, ticker: str) -> dict:
        try:
            quote = self.client.quote(ticker)
            # Fetch basic financials / company profile for context
            profile = self.client.company_profile2(symbol=ticker)
            
            price = quote.get('c', 0.0)
            prev_close = quote.get('pc', 0.0)
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0

            return {
                "ticker": ticker,
                "price": price,
                "high": quote.get('h', 0.0),
                "low": quote.get('l', 0.0),
                "open": quote.get('o', 0.0),
                "prev_close": prev_close,
                "change_pct": change_pct,
                "market_cap": profile.get('marketCapitalization', 0.0),
                "industry": profile.get('finnhubIndustry', 'Unknown'),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
        except Exception as e:
            return {"error": str(e), "ticker": ticker, "price": 0.0, "change_pct": 0.0}

class Subconscious:
    """
    SIMULATOR / IMAGINATION LAYER:
    Continuously runs forward-looking mental simulations (Monte Carlo + Deductive Graphs)
    to generate 'what-if' scenarios without risking real-world execution.
    """
    def __init__(self, num_simulations: int = 100, horizon_steps: int = 5):
        self.num_simulations = num_simulations
        self.horizon_steps = horizon_steps

    def run_mental_simulation(self, current_price: float, volatility: float = 0.02) -> dict:
        """Runs forward trajectories to evaluate possible future states."""
        if current_price <= 0:
            return {"mean_projected": 0, "bull_ratio": 0.5, "trajectories": []}

        # Simulate geometric brownian motion steps
        dt = 1
        drift = 0.0005
        
        simulations = np.zeros((self.num_simulations, self.horizon_steps))
        simulations[:, 0] = current_price

        for t in range(1, self.horizon_steps):
            shock = np.random.normal(0, 1, self.num_simulations)
            simulations[:, t] = simulations[:, t-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * shock)

        final_prices = simulations[:, -1]
        bullish_outcomes = np.sum(final_prices > current_price)
        
        return {
            "mean_projected": float(np.mean(final_prices)),
            "max_potential": float(np.max(final_prices)),
            "min_potential": float(np.min(final_prices)),
            "bull_ratio": float(bullish_outcomes / self.num_simulations),
            "trajectories": simulations.tolist()
        }

    def build_causal_knowledge_graph(self, market_data: dict, sim_data: dict) -> nx.DiGraph:
        """Constructs a graph mapping causal chains between Market State, Subconscious Scenarios, and Mind Actions."""
        G = nx.DiGraph()
        
        ticker = market_data.get("ticker", "ASSET")
        price = market_data.get("price", 0)
        bull_ratio = sim_data.get("bull_ratio", 0.5)

        # Nodes
        G.add_node(ticker, type="Asset", price=price)
        G.add_node("Price_Movement", change=market_data.get("change_pct", 0))
        G.add_node("Subconscious_Simulation", bull_probability=bull_ratio)
        G.add_node("Mind_Decision", state="Evaluating")

        # Causal Edges
        G.add_edge(ticker, "Price_Movement", relationship="exhibits")
        G.add_edge("Price_Movement", "Subconscious_Simulation", relationship="triggers_simulation")
        G.add_edge("Subconscious_Simulation", "Mind_Decision", relationship="informs_action")

        return G

class Mind:
    """
    DECISION ENGINE:
    Evaluates dynamic goals (Free Will concept: adaptable objective function based on internal risk tolerance).
    """
    def __init__(self, risk_appetite: float = 0.5):
        self.risk_appetite = risk_appetite  # Dynamic internal valuation state

    def evaluate(self, market_data: dict, sim_data: dict) -> dict:
        bull_ratio = sim_data.get("bull_ratio", 0.5)
        change_pct = market_data.get("change_pct", 0.0)

        # Dynamic Utility Threshold adjustment based on internal state
        buy_threshold = 0.65 - (self.risk_appetite * 0.15)
        sell_threshold = 0.35 + (self.risk_appetite * 0.15)

        if bull_ratio > buy_threshold and change_pct > -2.0:
            action = "ACCUMULATE / BUY"
            confidence = bull_ratio * 100
            conviction = "BULLISH"
        elif bull_ratio < sell_threshold or change_pct < -4.0:
            action = "LIQUIDATE / SHORT"
            confidence = (1 - bull_ratio) * 100
            conviction = "BEARISH"
        else:
            action = "HOLD / OBSERVE"
            confidence = 50.0
            conviction = "NEUTRAL"

        return {
            "action": action,
            "conviction": conviction,
            "confidence": round(confidence, 2),
            "applied_risk_appetite": self.risk_appetite,
            "reasoning": f"Simulated Bull Probability ({bull_ratio:.2f}) evaluated against dynamic threshold ({buy_threshold:.2f})."
        }
