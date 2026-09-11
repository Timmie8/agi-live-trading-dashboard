import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD

class StockAnalyzer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def get_stock_data(self, symbol: str, timeframe: str = "1d", period: str = "60d") -> pd.DataFrame:
        """
        Haalt candle data op via yfinance om Finnhub 403/API-limiet restricties te omzeilen.
        timeframe: '5m', '15m', '30m', '1d'
        """
        try:
            ticker = yf.Ticker(symbol)
            # Pas period aan op basis van interval ivm yfinance limieten
            if timeframe in ['5m', '15m', '30m']:
                period = '7d'
            
            df = ticker.history(period=period, interval=timeframe)

            if df.empty:
                return pd.DataFrame()

            df = df.reset_index()
            
            # Kolomnamen uniform maken
            time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
            df = df.rename(columns={time_col: 'Timestamp'})
            
            # Tijdszone verwijderen indien aanwezig voor strakke verwerking
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

        # 5. Kruisingen
        df['STO_Cross_Up'] = (df['Stoch_K'] > df['Stoch_D']) & (df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1))
        df['MACD_Cross_Up'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))

        return df

    def evaluate_signals(self, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 2:
            return {"status": "Geen data beschikbaar"}

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        sto_bullish = latest['Stoch_K'] > latest['Stoch_D'] and prev['Stoch_K'] <= prev['Stoch_D']
        sto_status = "BULLISH CROSS" if sto_bullish else ("BULLISH" if latest['Stoch_K'] > latest['Stoch_D'] else "BEARISH")

        macd_bullish = latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']
        macd_status = "BULLISH CROSS" if macd_bullish else ("BULLISH" if latest['MACD'] > latest['MACD_Signal'] else "BEARISH")

        rsi_overbought = latest['RSI'] > 70
        vol_avg = latest['Vol_SMA20'] if not pd.isna(latest['Vol_SMA20']) else 1
        vol_strong = latest['Volume'] > vol_avg * 1.5
        price_change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

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
            "RSI": round(latest['RSI'], 1) if not pd.isna(latest['RSI']) else 0,
            "STO_Status": sto_status,
            "MACD_Status": macd_status,
            "RSI_Overbought": rsi_overbought,
            "Action": action,
            "Reasons": reasons
        }
