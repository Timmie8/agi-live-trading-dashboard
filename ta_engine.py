import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class StockAnalyzer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def get_stock_data(self, symbol: str, timeframe: str = "1d", period: str = "1y") -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            if timeframe in ['5m', '15m', '30m']:
                period = '1mo'
            
            df = ticker.history(period=period, interval=timeframe)

            if df.empty:
                return pd.DataFrame()

            df = df.reset_index()
            time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
            df = df.rename(columns={time_col: 'Timestamp'})
            
            if hasattr(df['Timestamp'].dt, 'tz_localize'):
                df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

            return self._calculate_indicators(df)
        except Exception as e:
            print(f"Fout bij ophalen data voor {symbol}: {e}")
            return pd.DataFrame()

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 26:
            return df

        # 1. RSI (14)
        rsi_ind = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi_ind.rsi()

        # 2. Slow Stochastic Oscillator
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

        # 3. MACD
        macd_ind = MACD(close=df['Close'])
        df['MACD'] = macd_ind.macd()
        df['MACD_Signal'] = macd_ind.macd_signal()
        df['MACD_Hist'] = macd_ind.macd_diff()

        # 4. Volume Features
        df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
        df['Vol_Ratio'] = df['Volume'] / (df['Vol_SMA20'] + 1e-9)

        # 5. Price Returns
        df['Return'] = df['Close'].pct_change()

        return df.dropna().reset_index(drop=True)

    def evaluate_signals(self, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 2:
            return {"status": "Geen data beschikbaar"}

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        sto_bullish = latest['Stoch_K'] > latest['Stoch_D'] and prev['Stoch_K'] <= prev['Stoch_D']
        sto_status = "BULLISH CROSS" if sto_bullish else ("BULLISH" if latest['Stoch_K'] > latest['Stoch_D'] else "BEARISH")

        macd_bullish = latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']
        macd_status = "BULLISH CROSS" if macd_bullish else ("BULLISH" if latest['MACD'] > latest['MACD_Signal'] else "BEARISH")

        # RSI Niveaus
        current_rsi = latest['RSI']
        prev_rsi = prev['RSI']
        
        rsi_crossed_55 = current_rsi > 55 and prev_rsi <= 55
        rsi_above_55 = current_rsi > 55
        rsi_overbought = current_rsi > 70

        vol_avg = latest['Vol_SMA20'] if not pd.isna(latest['Vol_SMA20']) else 1
        vol_strong = latest['Volume'] > vol_avg * 1.5
        price_change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

        score = 0
        reasons = []

        # 📈 NIEUW: RSI > 55 Signaal & Logica
        if rsi_crossed_55:
            score += 2
            reasons.append("🔥 **RSI (14) BREAKOUT**: RSI is zojuist boven de 55 gestegen! (Bullish Momentum)")
        elif rsi_above_55 and not rsi_overbought:
            score += 1
            reasons.append("✅ RSI (14) bevindt zich boven de 55 (Positieve Trend)")

        if sto_bullish or latest['Stoch_K'] > latest['Stoch_D']:
            score += 1
            reasons.append("STO wijst omhoog (Korte termijn momentum)")

        if macd_bullish or latest['MACD'] > latest['MACD_Signal']:
            score += 1
            reasons.append("MACD is bullish (Kruising/Trend omhoog)")

        if vol_strong and price_change_pct > 0:
            score += 1
            reasons.append("Hoge stijging ondersteund door STERK volume")
        elif vol_strong and abs(price_change_pct) < 1.0:
            reasons.append("⚠️ Waarschuwing: Hoog volume ZONDER prijsverandering!")

        if rsi_overbought:
            reasons.append("⚠️ RSI > 70 (Koers is overbought, kans op pullback)")

        # 🎯 GEBASSINEERD ADVIES
        if rsi_overbought:
            action = "AVOID / TAKE PROFIT (Overbought)"
        elif rsi_above_55 and score >= 2:
            action = "STRONG BUY / BULLISH"
        elif score >= 2:
            action = "BUY / WATCH"
        else:
            action = "HOLD / WAIT (RSI < 55)"

        return {
            "Price": latest['Close'],
            "Change_Pct": round(price_change_pct, 2),
            "Volume": latest['Volume'],
            "RSI": round(current_rsi, 1) if not pd.isna(current_rsi) else 0,
            "RSI_Cross_55": rsi_crossed_55,
            "RSI_Above_55": rsi_above_55,
            "STO_Status": sto_status,
            "MACD_Status": macd_status,
            "RSI_Overbought": rsi_overbought,
            "Action": action,
            "Reasons": reasons
        }

    def predict_ml_probability(self, df: pd.DataFrame, forecast_horizon: int = 3, use_grid_search: bool = True) -> dict:
        if df.empty or len(df) < 60:
            return {"up_prob": 50.0, "status": "Onvoldoende data voor ML", "best_params": {}}

        data = df.copy()
        data['Target'] = (data['Close'].shift(-forecast_horizon) > data['Close']).astype(int)

        feature_cols = ['RSI', 'Stoch_K', 'Stoch_D', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Vol_Ratio', 'Return']
        
        X = data[feature_cols].iloc[:-forecast_horizon]
        y = data['Target'].iloc[:-forecast_horizon]

        if len(X) < 40:
            return {"up_prob": 50.0, "status": "Onvoldoende trainingsdata", "best_params": {}}

        rf_base = RandomForestClassifier(random_state=42)

        if use_grid_search:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 8],
                'min_samples_split': [2, 5]
            }

            grid_search = GridSearchCV(
                estimator=rf_base,
                param_grid=param_grid,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(X, y)
            best_clf = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            best_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            best_clf.fit(X, y)
            best_params = {"default": True}

        latest_features = data[feature_cols].iloc[[-1]]
        probs = best_clf.predict_proba(latest_features)[0]
        
        up_probability = round(probs[1] * 100, 2)
        importances = dict(zip(feature_cols, [round(x, 3) for x in best_clf.feature_importances_]))

        return {
            "up_prob": up_probability,
            "down_prob": round(100 - up_probability, 2),
            "best_params": best_params,
            "feature_importances": importances,
            "status": "Succesvol"
        }
