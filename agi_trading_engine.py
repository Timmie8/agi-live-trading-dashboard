import finnhub
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from datetime import datetime, timedelta

class StockAnalyzer:
    def __init__(self, api_key: str):
        self.client = finnhub.Client(api_key=api_key)

    def get_stock_data(self, symbol: str, resolution: str = "D", days_back: int = 100) -> pd.DataFrame:
        """
        Haalt candle data op via Finnhub.
        resolution: '1', '5', '15', '30', '60', 'D', 'W', 'M'
        """
        to_time = int(datetime.now().timestamp())
        from_time = int((datetime.now() - timedelta(days=days_back)).timestamp())

        res = self.client.stock_candles(symbol, resolution, from_time, to_time)
        
        if res.get('s') != 'ok':
            return pd.DataFrame()

        df = pd.DataFrame({
            'Timestamp': pd.to_datetime(res['t'], unit='s'),
            'Open': res['o'],
            'High': res['h'],
            'Low': res['l'],
            'Close': res['c'],
            'Volume': res['v']
        })
        
        # Bereken de indicatoren volgens jouw instructies
        return self._calculate_indicators(df)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 26:
            return df

        # 1. RSI (14)
        rsi_ind = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi_ind.rsi()

        # 2. Slow Stochastic Oscillator (Slow-Sto)
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

        # 3. MACD (12, 26, 9)
        macd_ind = MACD(close=df['Close'])
        df['MACD'] = macd_ind.macd()
        df['MACD_Signal'] = macd_ind.macd_signal()
        df['MACD_Hist'] = macd_ind.macd_diff()

        # 4. Volume Analyse
        df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
        df['High_Volume'] = df['Volume'] > (df['Vol_SMA20'] * 1.5)

        # 5. Signalen genereren op de meest recente candle
        df['STO_Cross_Up'] = (df['Stoch_K'] > df['Stoch_D']) & (df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1))
        df['MACD_Cross_Up'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))

        return df

    def evaluate_signals(self, df: pd.DataFrame) -> dict:
        """Evalueert de geselecteerde regels uit je handleiding op het meest recente datapunt."""
        if df.empty or len(df) < 2:
            return {"status": "Geen data beschikbaar"}

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # STO kruising
        sto_bullish = latest['Stoch_K'] > latest['Stoch_D'] and prev['Stoch_K'] <= prev['Stoch_D']
        sto_status = "BULLISH CROSS" if sto_bullish else ("BULLISH" if latest['Stoch_K'] > latest['Stoch_D'] else "BEARISH")

        # MACD kruising
        macd_bullish = latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']
        macd_status = "BULLISH CROSS" if macd_bullish else ("BULLISH" if latest['MACD'] > latest['MACD_Signal'] else "BEARISH")

        # RSI Overbought Check
        rsi_overbought = latest['RSI'] > 70
        rsi_oversold = latest['RSI'] < 30

        # Volume Bevestiging
        vol_strong = latest['Volume'] > latest['Vol_SMA20'] * 1.5
        price_change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

        # Totale Conclusie
        score = 0
        reasons = []

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

        action = "BUY / WATCH" if score >= 2 and not rsi_overbought else ("AVOID / CAUTION" if rsi_overbought else "WAIT")

        return {
            "Price": latest['Close'],
            "Change_Pct": round(price_change_pct, 2),
            "Volume": latest['Volume'],
            "Volume_Avg": round(latest['Vol_SMA20'], 0),
            "RSI": round(latest['RSI'], 1),
            "STO_Status": sto_status,
            "MACD_Status": macd_status,
            "RSI_Overbought": rsi_overbought,
            "Action": action,
            "Reasons": reasons
        }
