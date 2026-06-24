# ============================================================
#  信号生成模块
#  逐 Bar 扫描历史数据，生成具体的买卖信号（供回测 & 实盘使用）
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from config import RISK_PARAMS
from indicators import add_all_indicators


@dataclass
class Signal:
    date:      pd.Timestamp
    symbol:    str
    action:    str          # "BUY" / "SELL" / "SHORT" / "COVER"
    price:     float
    stop_loss: float
    target:    float
    atr:       float
    reason:    str


# ─────────────────────────────────────────────
#  入场信号规则
# ─────────────────────────────────────────────

def _long_entry(row: pd.Series, prev: pd.Series) -> tuple[bool, str]:
    """
    多头入场条件（需同时满足）：
    1. 价格在 MA20 上方（趋势方向正确）
    2. MACD 金叉（DIF 上穿 DEA）或 RSI 从超卖回升 > 35
    3. 成交量 > 5日均量 1.1 倍（量能配合）
    4. ATR 在正常范围（不过度波动）
    """
    reasons = []

    # 条件 1：价格在 MA20 上方
    if row["close"] <= row.get("MA20", row["close"]):
        return False, ""
    reasons.append("价格>MA20")

    # 条件 2：MACD 金叉 或 RSI 超卖回升
    macd_cross = (
        pd.notna(row.get("DIF")) and pd.notna(prev.get("DIF"))
        and row["DIF"] > row.get("DEA", 0)
        and prev["DIF"] <= prev.get("DEA", 0)
    )
    rsi_bounce = (
        pd.notna(prev.get("RSI")) and prev["RSI"] < 35
        and pd.notna(row.get("RSI")) and row["RSI"] >= 35
    )
    if not (macd_cross or rsi_bounce):
        return False, ""
    reasons.append("MACD金叉" if macd_cross else "RSI超卖回升")

    # 条件 3：成交量
    vol    = row.get("volume", 0)
    volma5 = row.get("VOL_MA5", vol)
    if volma5 > 0 and vol < volma5 * 1.1:
        return False, ""
    reasons.append("量能放大")

    return True, "；".join(reasons)


def _short_entry(row: pd.Series, prev: pd.Series) -> tuple[bool, str]:
    """
    空头入场条件（对称于多头）：
    1. 价格在 MA20 下方
    2. MACD 死叉 或 RSI 从超买回落 < 65
    3. 成交量 > 5日均量 1.1 倍
    """
    reasons = []

    if row["close"] >= row.get("MA20", row["close"]):
        return False, ""
    reasons.append("价格<MA20")

    macd_cross = (
        pd.notna(row.get("DIF")) and pd.notna(prev.get("DIF"))
        and row["DIF"] < row.get("DEA", 0)
        and prev["DIF"] >= prev.get("DEA", 0)
    )
    rsi_fall = (
        pd.notna(prev.get("RSI")) and prev["RSI"] > 65
        and pd.notna(row.get("RSI")) and row["RSI"] <= 65
    )
    if not (macd_cross or rsi_fall):
        return False, ""
    reasons.append("MACD死叉" if macd_cross else "RSI超买回落")

    vol    = row.get("volume", 0)
    volma5 = row.get("VOL_MA5", vol)
    if volma5 > 0 and vol < volma5 * 1.1:
        return False, ""
    reasons.append("量能放大")

    return True, "；".join(reasons)


# ─────────────────────────────────────────────
#  出场信号规则
# ─────────────────────────────────────────────

def _long_exit(row: pd.Series, prev: pd.Series,
               entry: float, stop: float, target: float) -> tuple[bool, str]:
    """多头持仓出场检测。"""
    close = row["close"]

    if close <= stop:
        return True, f"止损触发({close:.1f}<={stop:.1f})"
    if close >= target:
        return True, f"止盈触发({close:.1f}>={target:.1f})"

    # MACD 死叉 + 价格跌破 MA5
    if (pd.notna(row.get("DIF")) and row["DIF"] < row.get("DEA", 0)
            and pd.notna(prev.get("DIF")) and prev["DIF"] >= prev.get("DEA", 0)):
        if close < row.get("MA5", close):
            return True, "MACD死叉+跌破MA5"

    return False, ""


def _short_exit(row: pd.Series, prev: pd.Series,
                entry: float, stop: float, target: float) -> tuple[bool, str]:
    """空头持仓出场检测。"""
    close = row["close"]

    if close >= stop:
        return True, f"止损触发({close:.1f}>={stop:.1f})"
    if close <= target:
        return True, f"止盈触发({close:.1f}<={target:.1f})"

    if (pd.notna(row.get("DIF")) and row["DIF"] > row.get("DEA", 0)
            and pd.notna(prev.get("DIF")) and prev["DIF"] <= prev.get("DEA", 0)):
        if close > row.get("MA5", close):
            return True, "MACD金叉+突破MA5"

    return False, ""


# ─────────────────────────────────────────────
#  逐 Bar 扫描，生成信号序列
# ─────────────────────────────────────────────

def generate_signals(symbol: str, df: pd.DataFrame) -> list[Signal]:
    """
    对单个品种的历史数据逐 Bar 扫描，返回所有信号。
    不含仓位管理（由 backtester 处理）。
    """
    df_ind = add_all_indicators(df)
    df_ind = df_ind.dropna(subset=["MA20", "RSI", "ATR"]).reset_index(drop=True)

    sl_mult = RISK_PARAMS["atr_stop_mult"]
    tp_mult = RISK_PARAMS["atr_target_mult"]

    signals: list[Signal] = []
    position = None   # None / "LONG" / "SHORT"
    entry_price = stop_loss = target = atr_at_entry = 0.0

    for i in range(1, len(df_ind)):
        row  = df_ind.iloc[i]
        prev = df_ind.iloc[i - 1]
        date  = row["date"]
        close = row["close"]
        atr   = row["ATR"]

        if position is None:
            ok, reason = _long_entry(row, prev)
            if ok:
                entry_price   = close
                atr_at_entry  = atr
                stop_loss     = close - sl_mult * atr
                target        = close + tp_mult * atr
                position      = "LONG"
                signals.append(Signal(date, symbol, "BUY", close, stop_loss, target, atr, reason))
                continue

            ok, reason = _short_entry(row, prev)
            if ok:
                entry_price  = close
                atr_at_entry = atr
                stop_loss    = close + sl_mult * atr
                target       = close - tp_mult * atr
                position     = "SHORT"
                signals.append(Signal(date, symbol, "SHORT", close, stop_loss, target, atr, reason))
                continue

        elif position == "LONG":
            ok, reason = _long_exit(row, prev, entry_price, stop_loss, target)
            if ok:
                signals.append(Signal(date, symbol, "SELL", close, 0, 0, atr, reason))
                position = None

        elif position == "SHORT":
            ok, reason = _short_exit(row, prev, entry_price, stop_loss, target)
            if ok:
                signals.append(Signal(date, symbol, "COVER", close, 0, 0, atr, reason))
                position = None

    return signals


def get_latest_signal(symbol: str, df: pd.DataFrame) -> Signal | None:
    """只返回最新一个信号（用于实盘展示）。"""
    signals = generate_signals(symbol, df)
    return signals[-1] if signals else None
