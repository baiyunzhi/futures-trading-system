# ============================================================
#  数据获取模块
#  优先从 akshare 拉取真实行情，失败时生成仿真数据供回测
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from config import ALL_SYMBOLS, DATA_PARAMS

logger = logging.getLogger(__name__)

CACHE_DIR = Path(DATA_PARAMS["cache_dir"])
CACHE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
#  akshare 获取
# ─────────────────────────────────────────────

def _fetch_from_akshare(symbol: str) -> pd.DataFrame | None:
    """尝试通过 akshare 获取期货主力合约日线数据。"""
    try:
        import akshare as ak
        df = ak.futures_zh_daily_sina(symbol=symbol)
        if df is None or df.empty:
            return None

        # 统一列名
        df = df.rename(columns={
            "日期": "date", "date": "date",
            "开盘价": "open",  "open": "open",
            "最高价": "high",  "high": "high",
            "最低价": "low",   "low": "low",
            "收盘价": "close", "close": "close",
            "成交量": "volume","volume": "volume",
            "持仓量": "open_interest",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"])
        return df[["date", "open", "high", "low", "close", "volume"]]

    except Exception as e:
        logger.warning(f"akshare 获取 {symbol} 失败: {e}")
        return None


# ─────────────────────────────────────────────
#  仿真数据生成（备用）
# ─────────────────────────────────────────────

# 各品种大致价格中枢（元/手）
PRICE_ANCHORS = {
    "RB0": 3800, "HC0": 3900, "I0": 800,  "J0": 2200, "JM0": 1700,
    "CU0": 68000,"AL0": 18500,"ZN0": 22000,"NI0":130000,
    "M0":  3200, "Y0":  8200, "C0":  2400, "SR0": 6200, "CF0": 15000,
}

def _generate_sample_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """生成带趋势+周期+随机噪声的仿真期货日线数据。"""
    np.random.seed(hash(symbol) % (2**31))

    end   = datetime.today()
    start = end - timedelta(days=days * 1.5)
    dates = pd.bdate_range(start, end)[-days:]   # 取最近 N 个交易日

    base_price = PRICE_ANCHORS.get(symbol, 5000)
    vol_pct    = 0.012   # 日波动率约 1.2%

    # 生成带趋势的收盘价序列
    trend   = np.random.choice([-1, 0, 1], p=[0.3, 0.2, 0.5])
    returns = (np.random.randn(days) * vol_pct
               + trend * vol_pct * 0.15                    # 趋势偏移
               + 0.02 * np.sin(np.arange(days) / 30))    # 30日周期

    closes  = base_price * np.cumprod(1 + returns)
    highs   = closes * (1 + np.abs(np.random.randn(days)) * 0.006)
    lows    = closes * (1 - np.abs(np.random.randn(days)) * 0.006)
    opens   = np.roll(closes, 1) * (1 + np.random.randn(days) * 0.002)
    opens[0] = base_price
    volumes = np.random.randint(50_000, 500_000, size=days).astype(float)

    return pd.DataFrame({
        "date":   dates,
        "open":   np.round(opens, 0),
        "high":   np.round(highs, 0),
        "low":    np.round(lows, 0),
        "close":  np.round(closes, 0),
        "volume": volumes,
    })


# ─────────────────────────────────────────────
#  缓存层
# ─────────────────────────────────────────────

def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.csv"


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < DATA_PARAMS["cache_hours"]


# ─────────────────────────────────────────────
#  主接口
# ─────────────────────────────────────────────

def get_data(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    """
    获取品种日线数据。
    返回 DataFrame：date / open / high / low / close / volume
    """
    path = _cache_path(symbol)

    if use_cache and _is_cache_valid(path):
        logger.info(f"[缓存] 读取 {symbol}")
        return pd.read_csv(path, parse_dates=["date"])

    logger.info(f"[拉取] {symbol} ({ALL_SYMBOLS.get(symbol, symbol)})")
    df = _fetch_from_akshare(symbol)

    if df is None or len(df) < 60:
        logger.warning(f"[仿真] {symbol} 使用仿真数据")
        df = _generate_sample_data(symbol, days=DATA_PARAMS["lookback_days"])
        df["is_simulated"] = True
    else:
        df["is_simulated"] = False

    df.to_csv(path, index=False)
    return df


def get_all_data(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """批量获取所有品种数据，返回 {symbol: DataFrame}。"""
    result = {}
    for symbol in ALL_SYMBOLS:
        try:
            result[symbol] = get_data(symbol, use_cache=use_cache)
        except Exception as e:
            logger.error(f"获取 {symbol} 数据异常: {e}")
    return result


def get_latest_prices(data: dict[str, pd.DataFrame]) -> dict[str, float]:
    """返回每个品种的最新收盘价。"""
    return {sym: df["close"].iloc[-1] for sym, df in data.items() if not df.empty}
