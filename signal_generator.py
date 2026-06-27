# ============================================================
#  信号生成模块
#  逐 Bar 扫描历史数据，生成具体的买卖信号（供回测 & 实盘使用）
# ============================================================

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass


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


def generate_signals(symbol: str, df: pd.DataFrame) -> list[Signal]:
    import strategy_structure
    return strategy_structure.generate_signals(symbol, df)


def get_latest_signal(symbol: str, df: pd.DataFrame) -> Signal | None:
    """只返回最新一个信号（用于实盘展示）。"""
    signals = generate_signals(symbol, df)
    return signals[-1] if signals else None
