def evaluate_signals(self, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 2:
            return {"status": "Geen data beschikbaar"}

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        sto_bullish = latest['Stoch_K'] > latest['Stoch_D'] and prev['Stoch_K'] <= prev['Stoch_D']
        sto_status = "BULLISH CROSS" if sto_bullish else ("BULLISH" if latest['Stoch_K'] > latest['Stoch_D'] else "BEARISH")

        macd_bullish = latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']
        macd_status = "BULLISH CROSS" if macd_bullish else ("BULLISH" if latest['MACD'] > latest['MACD_Signal'] else "BEARISH")

        # RSI Niveaus & 55 Breakout Logica
        current_rsi = latest['RSI']
        prev_rsi = prev['RSI']

        rsi_crossed_55 = current_rsi > 55 and prev_rsi <= 55
        rsi_above_55 = current_rsi > 55
        
        # NIEUW: Overbought alleen bij DALING boven de 70
        rsi_above_70 = current_rsi > 70
        rsi_falling = current_rsi < prev_rsi
        rsi_overbought_warning = rsi_above_70 and rsi_falling

        vol_avg = latest['Vol_SMA20'] if not pd.isna(latest['Vol_SMA20']) else 1
        vol_strong = latest['Volume'] > vol_avg * 1.5
        price_change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

        score = 0
        reasons = []

        if rsi_crossed_55:
            score += 2
            reasons.append("🔥 **RSI (14) BREAKOUT**: RSI is zojuist boven de 55 gestegen!")
        elif rsi_above_70 and not rsi_falling:
            score += 2
            reasons.append("🚀 RSI (14) > 70 en stijgt (Extreem Bullish Momentum)")
        elif rsi_above_55 and not rsi_overbought_warning:
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

        if rsi_overbought_warning:
            reasons.append("⚠️ RSI daalt vanaf > 70 (Momentum zwakt af, mogelijke top)")

        # Advies Logica Aangepast
        if rsi_overbought_warning:
            action = "AVOID / TAKE PROFIT"
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
            "RSI_Overbought_Warning": rsi_overbought_warning,
            "RSI_Stijgend_Boven_70": rsi_above_70 and not rsi_falling,
            "STO_Status": sto_status,
            "MACD_Status": macd_status,
            "Action": action,
            "Reasons": reasons
        }
