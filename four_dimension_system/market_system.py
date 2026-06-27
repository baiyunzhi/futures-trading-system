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
    "CU0": {"name": "铜", "multiplier": 5},
    "M0": {"name": "豆粕", "multiplier": 10},
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
class BetPlan:
    action: str
    direction: str
    grade: str
    trigger: str
    entry: float
    stop: float
    target_1r: float
    target_2r: float
    risk_pct: float
    lots: int
    invalidation: str
    reason: str


@dataclass(frozen=True)
class MarketReport:
    symbol: str
    name: str
    views: dict[str, TimeframeView]
    story: str
    bet_plan: BetPlan


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
    configs = {
        "RB0": {"base": 3450, "trend": -1.8, "amp": 18, "oi": 820000},
        "CU0": {"base": 78500, "trend": 42, "amp": 360, "oi": 610000},
        "M0": {"base": 3120, "trend": 1.2, "amp": 24, "oi": 940000},
    }

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
    summary = f"{timeframe}: {price.state}，{volume.state}，{open_interest.state}，{time.state}"
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


def build_bet_plan(symbol: str, views: dict[str, TimeframeView]) -> BetPlan:
    weekly = views["周线"]
    daily = views["日线"]
    hourly = views["小时线"]
    direction = "观望"
    if weekly.bias == daily.bias == "bull":
        direction = "多"
    elif weekly.bias == daily.bias == "bear":
        direction = "空"

    latest = hourly.close
    multiplier = SYMBOLS[symbol]["multiplier"]
    capital = 500000
    risk_pct = 0.0
    action = "WAIT"
    grade = "观察"
    reason = "周线、日线、小时线未形成同向确认"

    if direction == "多":
        trigger_price = max(daily.high, hourly.high)
        stop = min(daily.low, hourly.low)
        risk = max(0.01, trigger_price - stop)
        target_1r = trigger_price + risk
        target_2r = trigger_price + 2 * risk
        trigger = f"小时线收盘突破 {trigger_price:.2f}，且成交量不缩、持仓量不减"
        invalidation = f"跌回 {stop:.2f} 下方"
    elif direction == "空":
        trigger_price = min(daily.low, hourly.low)
        stop = max(daily.high, hourly.high)
        risk = max(0.01, stop - trigger_price)
        target_1r = trigger_price - risk
        target_2r = trigger_price - 2 * risk
        trigger = f"小时线收盘跌破 {trigger_price:.2f}，且成交量不缩、持仓量不减"
        invalidation = f"重新站回 {stop:.2f} 上方"
    else:
        return BetPlan("WAIT", direction, grade, "等待三周期方向一致", latest, 0, 0, 0, 0, 0, "方向未确认", reason)

    expected = "bull" if direction == "多" else "bear"
    supports = [
        hourly.volume.bias in (expected, "neutral"),
        hourly.open_interest.bias in (expected, "neutral"),
        daily.volume.bias in (expected, "neutral"),
        daily.open_interest.bias in (expected, "neutral"),
    ]
    strong_support = hourly.volume.bias == expected and hourly.open_interest.bias == expected

    if hourly.bias == expected and all(supports) and strong_support:
        action = "BET"
        grade = "A"
        risk_pct = 0.01
        reason = "周线、日线、小时线同向，且四维度支持当前方向"
    elif hourly.bias == expected and all(supports):
        action = "PLAN"
        grade = "B"
        risk_pct = 0.005
        reason = "价格三周期同向，成交量和持仓量未反对，等待放量或增仓后升级为下注"
    else:
        action = "WAIT"
        grade = "观察"
        risk_pct = 0.0
        reason = "周线和日线同向，但小时线尚未触发"

    risk_amount = capital * risk_pct
    lots = int(risk_amount / (risk * multiplier)) if risk > 0 else 0
    return BetPlan(
        action=action,
        direction=direction,
        grade=grade,
        trigger=trigger,
        entry=round(trigger_price, 2),
        stop=round(stop, 2),
        target_1r=round(target_1r, 2),
        target_2r=round(target_2r, 2),
        risk_pct=risk_pct,
        lots=max(0, lots),
        invalidation=invalidation,
        reason=reason,
    )


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
            plan = build_bet_plan(symbol, views)
            story = "；".join(views[name].summary for name in TIMEFRAMES)
            reports.append(MarketReport(symbol, meta["name"], views, story, plan))
    order = {"BET": 3, "PLAN": 2, "WAIT": 1}
    return sorted(reports, key=lambda item: (order.get(item.bet_plan.action, 0), item.bet_plan.grade), reverse=True)


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
