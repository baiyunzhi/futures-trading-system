from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sin
from pathlib import Path

import pandas as pd


TIMEFRAMES = ("周线", "日线", "小时线")

SYMBOLS = {
    "RB0": {"name": "螺纹钢", "multiplier": 10},
}


@dataclass(frozen=True)
class TimeframeView:
    timeframe: str
    summary: str
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class MarketReport:
    symbol: str
    name: str
    views: dict[str, TimeframeView]
    story: str


def load_or_create_data(path: Path) -> pd.DataFrame:
    if path.exists():
        return _normalize_data(pd.read_csv(path))
    df = create_sample_hourly_data()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return _normalize_data(df)


def create_sample_hourly_data() -> pd.DataFrame:
    rows: list[dict] = []
    start = datetime(2026, 3, 2, 9)
    hours = [9, 10, 11, 13, 14]
    configs = {"RB0": {"base": 3450, "trend": -1.8, "amp": 18, "oi": 820000}}

    for symbol, cfg in configs.items():
        step = 0
        for day in range(90):
            current = start + timedelta(days=day)
            if current.weekday() >= 5:
                continue
            for hour in hours:
                dt = current.replace(hour=hour)
                drift = cfg["trend"] * step / len(hours)
                wave = cfg["amp"] * sin(step / 7)
                base = cfg["base"] + drift + wave
                close = base + cfg["amp"] * 0.12 * sin(step / 3)
                open_ = base - cfg["amp"] * 0.08 * sin(step / 4)
                high = max(open_, close) + cfg["amp"] * (0.35 + 0.08 * abs(sin(step)))
                low = min(open_, close) - cfg["amp"] * (0.32 + 0.07 * abs(sin(step / 2)))
                volume = 50000 + abs(sin(step / 5)) * 70000 + (90000 if step % 37 == 0 else 0)
                oi = cfg["oi"] + step * (280 if cfg["trend"] > 0 else -180) + 6000 * sin(step / 11)
                rows.append({
                    "datetime": dt.isoformat(sep=" "),
                    "symbol": symbol,
                    "open": round(open_, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "volume": round(volume, 0),
                    "open_interest": round(max(1, oi), 0),
                })
                step += 1
    return pd.DataFrame(rows)


def _normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    required = ["datetime", "symbol", "open", "high", "low", "close", "volume", "open_interest"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"数据缺少字段: {missing}")
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "open_interest"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required)
    out = out[(out["high"] >= out[["open", "close", "low"]].max(axis=1))]
    out = out[(out["low"] <= out[["open", "close", "high"]].min(axis=1))]
    return out.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def build_timeframes(symbol_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    indexed = symbol_df.set_index("datetime").sort_index()
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "open_interest": "last",
    }
    return {
        "周线": indexed.resample("W-FRI").agg(agg).dropna().reset_index(),
        "日线": indexed.resample("1D").agg(agg).dropna().reset_index(),
        "小时线": indexed.reset_index(),
    }


def combine_view(timeframe: str, frame: pd.DataFrame) -> TimeframeView:
    latest = frame.iloc[-1]
    summary = build_objective_summary(timeframe, frame)
    return TimeframeView(
        timeframe=timeframe,
        summary=summary,
        high=round(float(latest["high"]), 2),
        low=round(float(latest["low"]), 2),
        close=round(float(latest["close"]), 2),
    )


def build_objective_summary(timeframe: str, frame: pd.DataFrame) -> str:
    latest = frame.iloc[-1]
    prev = frame.iloc[-2] if len(frame) >= 2 else latest
    tail = frame.tail(12)
    latest_high = float(latest["high"])
    latest_low = float(latest["low"])
    latest_open = float(latest["open"])
    latest_close = float(latest["close"])
    period_high_idx = tail["high"].idxmax()
    period_low_idx = tail["low"].idxmin()
    period_high_row = tail.loc[period_high_idx]
    period_low_row = tail.loc[period_low_idx]
    price_change = latest_close - float(prev["close"])
    volume_change = float(latest["volume"] - prev["volume"])
    oi_change = float(latest["open_interest"] - prev["open_interest"])
    attitude = describe_attitude(price_change, volume_change, oi_change)
    return (
        f"{timeframe}独立观察：最新K线时间 {format_dt(latest['datetime'])}，"
        f"开盘 {latest_open:.2f}，高点 {latest_high:.2f}，低点 {latest_low:.2f}，收盘 {latest_close:.2f}。"
        f"最近12根K线最高点 {float(period_high_row['high']):.2f}，时间 {format_dt(period_high_row['datetime'])}；"
        f"最低点 {float(period_low_row['low']):.2f}，时间 {format_dt(period_low_row['datetime'])}。"
        f"与上一根K线相比，收盘变化 {price_change:+.2f}，成交量变化 {volume_change:+.0f}，"
        f"持仓量变化 {oi_change:+.0f}。{attitude}"
    )


def describe_attitude(price_change: float, volume_change: float, oi_change: float) -> str:
    if price_change > 0 and volume_change > 0 and oi_change > 0:
        return "价格上行，同时成交量和持仓量增加，当前K线显示多方参与增加。"
    if price_change < 0 and volume_change > 0 and oi_change > 0:
        return "价格下行，同时成交量和持仓量增加，当前K线显示空方参与增加。"
    if price_change > 0 and oi_change < 0:
        return "价格上行但持仓量减少，当前K线显示上涨过程中有持仓退出。"
    if price_change < 0 and oi_change < 0:
        return "价格下行且持仓量减少，当前K线显示下跌过程中有持仓退出。"
    if volume_change > 0:
        return "成交量增加，当前K线参与度比上一根提高。"
    if volume_change < 0:
        return "成交量减少，当前K线参与度比上一根降低。"
    return "成交量和持仓量较上一根K线没有明显变化。"


def format_dt(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")


def analyze_market(data: pd.DataFrame) -> list[MarketReport]:
    reports: list[MarketReport] = []
    clean = _normalize_data(data)
    for symbol, meta in SYMBOLS.items():
        symbol_df = clean[clean["symbol"] == symbol]
        if symbol_df.empty:
            continue
        frames = build_timeframes(symbol_df)
        views = {name: combine_view(name, frame) for name, frame in frames.items() if len(frame) >= 5}
        if set(TIMEFRAMES).issubset(views):
            story = "；".join(views[name].summary for name in TIMEFRAMES)
            reports.append(MarketReport(symbol, meta["name"], views, story))
    return reports
