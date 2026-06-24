# ============================================================
#  技术指标计算模块
#  所有指标直接在 DataFrame 上追加列，返回同一个 DataFrame
# ============================================================

import numpy as np
import pandas as pd
from config import INDICATOR_PARAMS


def add_ma(df: pd.DataFrame) -> pd.DataFrame:
    """添加多周期均线 MA5 / MA10 / MA20 / MA60。"""
    for period in INDICATOR_PARAMS["MA"]:
        df[f"MA{period}"] = df["close"].rolling(period).mean()
    return df


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """添加 RSI(14)。"""
    n = INDICATOR_PARAMS["RSI"]
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """添加 MACD DIF / DEA / HIST。"""
    fast, slow, signal = INDICATOR_PARAMS["MACD"]
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["DIF"]  = ema_fast - ema_slow
    df["DEA"]  = df["DIF"].ewm(span=signal, adjust=False).mean()
    df["HIST"] = (df["DIF"] - df["DEA"]) * 2
    return df


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


def add_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    """添加布林带 BB_MID / BB_UPPER / BB_LOWER。"""
    period, mult = INDICATOR_PARAMS["BB"]
    mid  = df["close"].rolling(period).mean()
    std  = df["close"].rolling(period).std()
    df["BB_MID"]   = mid
    df["BB_UPPER"] = mid + mult * std
    df["BB_LOWER"] = mid - mult * std
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / mid  # 带宽 / 中轨
    return df


def add_volume_ma(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """添加成交量均线 VOL_MA5，用于量价配合判断。"""
    df[f"VOL_MA{period}"] = df["volume"].rolling(period).mean()
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """添加 ADX（平均趋向指标），衡量趋势强度。"""
    high, low, close_prev = df["high"], df["low"], df["close"].shift(1)

    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low  - close_prev).abs(),
    ], axis=1).max(axis=1)

    atr14     = tr.rolling(period).mean()
    plus_di   = 100 * pd.Series(plus_dm,  index=df.index).rolling(period).mean() / atr14
    minus_di  = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr14
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    df["+DI"]  = plus_di
    df["-DI"]  = minus_di
    df["ADX"]  = dx.rolling(period).mean()
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """一次性添加全部指标，是对外主接口。"""
    df = df.copy()
    df = add_ma(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_atr(df)
    df = add_bollinger(df)
    df = add_volume_ma(df)
    df = add_adx(df)
    return df


def get_latest_row(df: pd.DataFrame) -> pd.Series:
    """取最新一行有效数据。"""
    return df.dropna(subset=["MA20", "RSI", "ATR"]).iloc[-1]
