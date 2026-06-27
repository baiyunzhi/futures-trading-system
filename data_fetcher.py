# ============================================================
#  数据获取模块
#  优先从 akshare 拉取真实行情，失败时生成仿真数据供回测
# ============================================================

import os
import json
import time
import logging
import hashlib
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

# akshare 可能返回的列名映射（兼容多版本）
_COL_MAP = {
    # 中文列名（老版本）
    "日期":  "date", "开盘价": "open",  "最高价": "high",
    "最低价": "low",  "收盘价": "close", "成交量": "volume",
    "持仓量": "open_interest",
    # 英文列名（新版本）
    "date":  "date", "open":  "open",  "high":  "high",
    "low":   "low",  "close": "close", "volume": "volume",
    "open_interest": "open_interest",
    "hold": "open_interest",
    # sina 格式
    "settle": "settle", "pre_settle": "pre_settle",
}


def _fetch_from_akshare(symbol: str) -> pd.DataFrame | None:
    """尝试通过 akshare 获取期货主力合约日线数据（兼容多版本列名）。"""
    try:
        import akshare as ak

        # 尝试多个接口，按优先级排列
        df = None
        errors = []

        for fn_name, kwargs in [
            ("futures_zh_daily_sina",    {"symbol": symbol}),
            ("futures_main_sina",        {"symbol": symbol}),
            ("futures_zh_spot",          {"symbol": symbol, "market": "CF"}),
        ]:
            try:
                fn = getattr(ak, fn_name, None)
                if fn is None:
                    continue
                df = fn(**kwargs)
                if df is not None and not df.empty:
                    logger.debug(f"[akshare] {symbol} 使用接口 {fn_name}")
                    break
                df = None
            except Exception as e:
                errors.append(f"{fn_name}: {e}")
                df = None

        if df is None or df.empty:
            logger.warning(f"akshare 获取 {symbol} 失败: {'; '.join(errors)}")
            return None

        # 统一列名（自动映射，容错大小写）
        rename = {}
        for col in df.columns:
            mapped = _COL_MAP.get(col) or _COL_MAP.get(col.lower())
            if mapped:
                rename[col] = mapped
        df = df.rename(columns=rename)

        # 确保 date 列存在
        if "date" not in df.columns:
            for c in df.columns:
                if "date" in c.lower() or "时间" in c or "日期" in c:
                    df = df.rename(columns={c: "date"})
                    break

        if "date" not in df.columns:
            logger.warning(f"akshare {symbol} 无法识别日期列，列名: {list(df.columns)}")
            return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 保留可用列
        keep = ["date", "open", "high", "low", "close", "volume"]
        if "open_interest" in df.columns:
            keep.append("open_interest")
        return _validate_ohlcv(df[[c for c in keep if c in df.columns]], symbol)

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

def _stable_seed(symbol: str) -> int:
    return int(hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:8], 16)


def _generate_sample_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """生成带趋势+周期+随机噪声的仿真期货日线数据。"""
    np.random.seed(_stable_seed(symbol))

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
    opens   = np.roll(closes, 1) * (1 + np.random.randn(days) * 0.002)
    opens[0] = base_price
    base_high = np.maximum(opens, closes)
    base_low  = np.minimum(opens, closes)
    highs   = base_high * (1 + np.abs(np.random.randn(days)) * 0.006)
    lows    = base_low * (1 - np.abs(np.random.randn(days)) * 0.006)
    volumes = np.random.randint(50_000, 500_000, size=days).astype(float)

    return pd.DataFrame({
        "date":   dates,
        "open":   np.round(opens, 0),
        "high":   np.round(highs, 0),
        "low":    np.round(lows, 0),
        "close":  np.round(closes, 0),
        "volume": volumes,
    })


def _validate_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"{symbol} 行情缺少字段: {missing}")
        return None

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "open_interest" in out.columns:
        out["open_interest"] = pd.to_numeric(out["open_interest"], errors="coerce")

    before = len(out)
    out = out.dropna(subset=required)
    out = out[(out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0)]
    out = out[out["volume"] >= 0]
    out = out[(out["high"] >= out[["open", "close", "low"]].max(axis=1)) &
              (out["low"] <= out[["open", "close", "high"]].min(axis=1))]
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        logger.warning(f"{symbol} 数据质量过滤 {dropped} 行")
    if len(out) < 60:
        return None
    return out


# ─────────────────────────────────────────────
#  缓存层
# ─────────────────────────────────────────────

def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.csv"


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


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
    返回 DataFrame：date / open / high / low / close / volume / open_interest(若数据源提供)
    """
    path = _cache_path(symbol)

    if use_cache and _is_cache_valid(path):
        logger.info(f"[缓存] 读取 {symbol}")
        cached = pd.read_csv(path, parse_dates=["date"])
        validated = _validate_ohlcv(cached, symbol)
        if validated is not None:
            if "is_simulated" in cached.columns:
                is_simulated = _as_bool(cached["is_simulated"].iloc[-1])
                if is_simulated and len(validated) < int(DATA_PARAMS["lookback_days"] * 0.8):
                    logger.warning(f"[缓存失效] {symbol} 仿真缓存长度不足，重新生成")
                else:
                    validated["is_simulated"] = is_simulated
                    return validated
            else:
                return validated
        logger.warning(f"[缓存失效] {symbol} 缓存数据质量不合格，重新拉取")

    logger.info(f"[拉取] {symbol} ({ALL_SYMBOLS.get(symbol, symbol)})")
    df = _fetch_from_akshare(symbol)

    if df is None or len(df) < 60:
        if path.exists():
            cached = pd.read_csv(path, parse_dates=["date"])
            validated = _validate_ohlcv(cached, symbol)
            cached_simulated = _as_bool(cached["is_simulated"].iloc[-1]) if "is_simulated" in cached.columns else False
            if validated is not None and not cached_simulated and len(validated) >= 60:
                logger.warning(f"[旧缓存] {symbol} 拉取失败，保留上一份真实行情缓存")
                validated["is_simulated"] = False
                return validated
        logger.warning(f"[仿真] {symbol} 使用仿真数据")
        df = _generate_sample_data(symbol, days=DATA_PARAMS["lookback_days"])
        df = _validate_ohlcv(df, symbol)
        if df is None:
            raise ValueError(f"{symbol} 仿真数据生成失败")
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
