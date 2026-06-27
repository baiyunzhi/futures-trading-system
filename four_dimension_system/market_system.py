from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sin
from pathlib import Path

import pandas as pd


TIMEFRAMES = ("周线", "日线", "小时线")
DIMENSIONS = ("价格", "成交量", "持仓量", "时间")

SYMBOLS = {
    "RB0": {"name": "螺纹钢", "multiplier": 10},
}


@dataclass(frozen=True)
class DimensionView:
    state: str
    bias: str
    evidence: str


@dataclass(frozen=True)
class TimeframeView:
    timeframe: str
    price: DimensionView
    volume: DimensionView
    open_interest: DimensionView
    time: DimensionView
    summary: str
    bias: str
    score: int
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


def analyze_price(frame: pd.DataFrame) -> DimensionView:
    latest = frame.iloc[-1]
    history = frame.iloc[:-1].tail(20)
    if history.empty:
        return DimensionView("数据不足", "neutral", "历史K线不足")
    prev_high = float(history["high"].max())
    prev_low = float(history["low"].min())
    close = float(latest["close"])
    if close > prev_high:
        return DimensionView("向上突破", "bull", f"收盘 {close:.2f} 高于前区间高点 {prev_high:.2f}")
    if close < prev_low:
        return DimensionView("向下跌破", "bear", f"收盘 {close:.2f} 低于前区间低点 {prev_low:.2f}")
    width = prev_high - prev_low
    pos = (close - prev_low) / width if width > 0 else 0.5
    if pos >= 0.67:
        return DimensionView("区间上沿", "bull", f"价格位于近20根区间上部 {pos:.0%}")
    if pos <= 0.33:
        return DimensionView("区间下沿", "bear", f"价格位于近20根区间下部 {pos:.0%}")
    return DimensionView("区间中部", "neutral", f"价格位于近20根区间中部 {pos:.0%}")


def analyze_volume(frame: pd.DataFrame) -> DimensionView:
    latest = frame.iloc[-1]
    history = frame.iloc[:-1].tail(20)
    if history.empty:
        return DimensionView("数据不足", "neutral", "历史成交量不足")
    current = float(latest["volume"])
    median = float(history["volume"].median())
    ratio = current / median if median > 0 else 1.0
    change = float(latest["close"] - latest["open"])
    if ratio >= 1.4 and change > 0:
        return DimensionView("放量上涨", "bull", f"成交量为中位数 {ratio:.2f} 倍")
    if ratio >= 1.4 and change < 0:
        return DimensionView("放量下跌", "bear", f"成交量为中位数 {ratio:.2f} 倍")
    if ratio <= 0.75:
        return DimensionView("缩量", "neutral", f"成交量为中位数 {ratio:.2f} 倍")
    return DimensionView("量能平稳", "neutral", f"成交量为中位数 {ratio:.2f} 倍")


def analyze_open_interest(frame: pd.DataFrame) -> DimensionView:
    if len(frame) < 2:
        return DimensionView("数据不足", "neutral", "历史持仓量不足")
    latest = frame.iloc[-1]
    prev = frame.iloc[-2]
    oi_now = float(latest["open_interest"])
    oi_prev = float(prev["open_interest"])
    oi_change = (oi_now - oi_prev) / oi_prev * 100 if oi_prev > 0 else 0.0
    price_change = float(latest["close"] - prev["close"])
    if oi_change > 0.5 and price_change > 0:
        return DimensionView("增仓上涨", "bull", f"持仓量变化 {oi_change:+.2f}%")
    if oi_change > 0.5 and price_change < 0:
        return DimensionView("增仓下跌", "bear", f"持仓量变化 {oi_change:+.2f}%")
    if oi_change < -0.5:
        return DimensionView("减仓", "neutral", f"持仓量变化 {oi_change:+.2f}%")
    return DimensionView("持仓平稳", "neutral", f"持仓量变化 {oi_change:+.2f}%")


def analyze_time(frame: pd.DataFrame, timeframe: str) -> DimensionView:
    latest = frame.iloc[-1]
    history = frame.iloc[:-1].tail(20)
    if history.empty:
        return DimensionView("数据不足", "neutral", "历史时间样本不足")
    current_range = float(latest["high"] - latest["low"])
    median_range = float((history["high"] - history["low"]).median())
    ratio = current_range / median_range if median_range > 0 else 1.0
    if ratio >= 1.35:
        return DimensionView("波动扩张", "neutral", f"{timeframe}波动为中位数 {ratio:.2f} 倍")
    if ratio <= 0.75:
        return DimensionView("波动收缩", "neutral", f"{timeframe}波动为中位数 {ratio:.2f} 倍")
    return DimensionView("节奏平稳", "neutral", f"{timeframe}波动为中位数 {ratio:.2f} 倍")


def combine_view(timeframe: str, frame: pd.DataFrame) -> TimeframeView:
    price = analyze_price(frame)
    volume = analyze_volume(frame)
    open_interest = analyze_open_interest(frame)
    time = analyze_time(frame, timeframe)
    score = _bias_score([price.bias, volume.bias, open_interest.bias])
    bias = price.bias
    latest = frame.iloc[-1]
    summary = build_objective_summary(timeframe, frame, price, volume, open_interest, time)
    return TimeframeView(
        timeframe=timeframe,
        price=price,
        volume=volume,
        open_interest=open_interest,
        time=time,
        summary=summary,
        bias=bias,
        score=score,
        high=round(float(latest["high"]), 2),
        low=round(float(latest["low"]), 2),
        close=round(float(latest["close"]), 2),
    )


def _bias_score(items: list[str]) -> int:
    score = 0
    for item in items:
        if item == "bull":
            score += 1
        elif item == "bear":
            score -= 1
    return score


def build_objective_summary(
    timeframe: str,
    frame: pd.DataFrame,
    price: DimensionView,
    volume: DimensionView,
    open_interest: DimensionView,
    time: DimensionView,
) -> str:
    latest = frame.iloc[-1]
    history = frame.iloc[:-1].tail(20)
    recent_high = float(history["high"].max()) if not history.empty else float(latest["high"])
    recent_low = float(history["low"].min()) if not history.empty else float(latest["low"])
    latest_high = float(latest["high"])
    latest_low = float(latest["low"])
    latest_close = float(latest["close"])
    latest_range = latest_high - latest_low
    volume_text = _attitude_text(volume, open_interest)
    return (
        f"{timeframe}独立观察：最新收盘 {latest_close:.2f}，K线高点 {latest_high:.2f}、低点 {latest_low:.2f}。"
        f"近20根高点 {recent_high:.2f}、低点 {recent_low:.2f}，当前价格表现为{price.state}。"
        f"本周期单根波动 {latest_range:.2f}，时间节奏为{time.state}。"
        f"成交量表现为{volume.state}，持仓量表现为{open_interest.state}。"
        f"{volume_text}"
    )


def _attitude_text(volume: DimensionView, open_interest: DimensionView) -> str:
    if volume.bias == "bull" and open_interest.bias == "bull":
        return "成交量和持仓量共同显示多方主动增加，行情上行波动的持续性较强。"
    if volume.bias == "bear" and open_interest.bias == "bear":
        return "成交量和持仓量共同显示空方主动增加，行情下行波动的持续性较强。"
    if open_interest.state == "减仓":
        return "持仓量下降，说明当前波动更多来自离场或减仓，持续性需要降低评价。"
    if volume.state == "缩量":
        return "成交量收缩，说明当前价格波动缺少主动成交推动。"
    return "成交量和持仓量没有形成明显单边态度，当前波动暂未显示持续性增强。"


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


def sparkline_svg(frame: pd.DataFrame, width: int = 520, height: int = 120) -> str:
    tail = frame.tail(80)
    if tail.empty:
        return ""
    closes = tail["close"].astype(float).tolist()
    lo = min(closes)
    hi = max(closes)
    span = hi - lo if hi > lo else 1
    points = []
    for i, close in enumerate(closes):
        x = i / max(1, len(closes) - 1) * width
        y = height - (close - lo) / span * height
        points.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg viewBox="0 0 {width} {height}" class="spark">'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#4cc9f0" stroke-width="2"/>'
        f'<line x1="0" y1="{height - 1}" x2="{width}" y2="{height - 1}" stroke="#2a2f3a"/>'
        f"</svg>"
    )
