from __future__ import annotations

import pandas as pd

from config import RISK_PARAMS
from indicators import add_all_indicators
from signal_generator import Signal


LOOKBACK = 20
TREND_LOOKBACK = 60
HOLD_MAX_BARS = 20


def _brackets(close: float, atr: float, direction: str, high_level: float, low_level: float) -> tuple[float, float]:
    stop_mult = RISK_PARAMS["atr_stop_mult"]
    target_mult = RISK_PARAMS["atr_target_mult"]
    if direction == "LONG":
        stop = min(low_level - 0.5 * atr, close - stop_mult * atr)
        target = close + max((close - stop) * 1.8, target_mult * atr)
    else:
        stop = max(high_level + 0.5 * atr, close + stop_mult * atr)
        target = close - max((stop - close) * 1.8, target_mult * atr)
    return round(float(stop), 4), round(float(target), 4)


def _structure_context(df: pd.DataFrame, i: int) -> dict:
    prior = df.iloc[:i]
    recent = prior.tail(LOOKBACK)
    trend = prior.tail(TREND_LOOKBACK)
    recent_high = float(recent["high"].max())
    recent_low = float(recent["low"].min())
    range_pct = (recent_high - recent_low) / recent_low * 100 if recent_low > 0 else 0.0

    half = max(LOOKBACK, len(trend) // 2)
    old = trend.iloc[:half]
    new = trend.iloc[half:]
    if old.empty or new.empty:
        trend_state = "RANGE"
    else:
        high_up = float(new["high"].max()) > float(old["high"].max())
        low_up = float(new["low"].min()) > float(old["low"].min())
        high_dn = float(new["high"].max()) < float(old["high"].max())
        low_dn = float(new["low"].min()) < float(old["low"].min())
        if high_up and low_up:
            trend_state = "UPTREND"
        elif high_dn and low_dn:
            trend_state = "DOWNTREND"
        else:
            trend_state = "RANGE"

    return {
        "recent_high": recent_high,
        "recent_low": recent_low,
        "range_pct": range_pct,
        "trend": trend_state,
    }


def generate_signals(symbol: str, df: pd.DataFrame, allowed_regime: str | None = None) -> list[Signal]:
    df_ind = add_all_indicators(df).dropna(subset=["ATR"]).reset_index(drop=True)
    signals: list[Signal] = []
    position: str | None = None
    entry_bar = 0
    stop_loss = target = 0.0

    for i in range(max(TREND_LOOKBACK, LOOKBACK) + 1, len(df_ind)):
        row = df_ind.iloc[i]
        prev = df_ind.iloc[i - 1]
        date = row["date"]
        close = float(row["close"])
        prev_close = float(prev["close"])
        atr = float(row["ATR"])
        ctx = _structure_context(df_ind, i)
        recent_high = ctx["recent_high"]
        recent_low = ctx["recent_low"]
        range_pct = ctx["range_pct"]
        trend = ctx["trend"]

        breakout_up = close > recent_high + 0.1 * atr
        breakout_dn = close < recent_low - 0.1 * atr
        near_support = close <= recent_low + 1.2 * atr
        near_resistance = close >= recent_high - 1.2 * atr

        if breakout_up or breakout_dn:
            regime = "BREAKOUT"
        elif trend in ("UPTREND", "DOWNTREND"):
            regime = "TREND"
        else:
            regime = "RANGE"
        if allowed_regime is not None and regime != allowed_regime:
            continue

        if position is None:
            if breakout_up:
                stop_loss, target = _brackets(close, atr, "LONG", recent_high, recent_low)
                position = "LONG"
                entry_bar = i
                signals.append(Signal(date, symbol, "BUY", close, stop_loss, target, atr, "结构向上突破"))
            elif breakout_dn:
                stop_loss, target = _brackets(close, atr, "SHORT", recent_high, recent_low)
                position = "SHORT"
                entry_bar = i
                signals.append(Signal(date, symbol, "SHORT", close, stop_loss, target, atr, "结构向下突破"))
            continue

        hold_bars = i - entry_bar
        if position == "LONG":
            if close <= stop_loss:
                signals.append(Signal(date, symbol, "SELL", close, 0, 0, atr, "结构止损"))
                position = None
            elif close >= target:
                signals.append(Signal(date, symbol, "SELL", close, 0, 0, atr, "结构止盈"))
                position = None
            elif breakout_dn or hold_bars >= HOLD_MAX_BARS:
                signals.append(Signal(date, symbol, "SELL", close, 0, 0, atr, "结构转弱或持仓超时"))
                position = None
        elif position == "SHORT":
            if close >= stop_loss:
                signals.append(Signal(date, symbol, "COVER", close, 0, 0, atr, "结构止损"))
                position = None
            elif close <= target:
                signals.append(Signal(date, symbol, "COVER", close, 0, 0, atr, "结构止盈"))
                position = None
            elif breakout_up or hold_bars >= HOLD_MAX_BARS:
                signals.append(Signal(date, symbol, "COVER", close, 0, 0, atr, "结构转强或持仓超时"))
                position = None

    return signals


def generate_trend_signals(symbol: str, df: pd.DataFrame) -> list[Signal]:
    return generate_signals(symbol, df, allowed_regime="TREND")


def generate_breakout_signals(symbol: str, df: pd.DataFrame) -> list[Signal]:
    return generate_signals(symbol, df, allowed_regime="BREAKOUT")


def generate_range_signals(symbol: str, df: pd.DataFrame) -> list[Signal]:
    return generate_signals(symbol, df, allowed_regime="RANGE")
