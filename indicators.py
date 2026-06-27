# ============================================================
#  技术指标计算模块
#  所有指标直接在 DataFrame 上追加列，返回同一个 DataFrame
# ============================================================

import numpy as np
import pandas as pd
from config import INDICATOR_PARAMS


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    """添加 ATR(14)，用于止损计算。"""
    n = INDICATOR_PARAMS["ATR"]
    high, low, close_prev = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low  - close_prev).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(n).mean()
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加交易系统仍实际使用的必要指标。"""
    df = df.copy()
    df = add_atr(df)
    return df


def get_latest_row(df: pd.DataFrame) -> pd.Series:
    """取最新一行有效数据。"""
    valid = df.dropna(subset=["ATR"])
    if valid.empty:
        raise ValueError("指标有效数据为空，至少需要 ATR 完整的行情序列")
    return valid.iloc[-1]
