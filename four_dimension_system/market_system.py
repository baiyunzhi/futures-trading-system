from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TIMEFRAMES = ("日线", "周线")

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
    if not path.exists():
        raise FileNotFoundError(f"缺少螺纹钢最近两个月数据文件: {path}")
    return _normalize_data(pd.read_csv(path))


def _normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"date": "datetime"})
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
        "日线": indexed.reset_index(),
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
    first = frame.iloc[0]
    max_volume_row = frame.loc[frame["volume"].idxmax()]
    structure = describe_price_structure(frame)
    max_volume_story = describe_max_volume_area(frame, max_volume_row)
    oi_story = describe_open_interest(frame)
    latest_high = float(latest["high"])
    latest_low = float(latest["low"])
    latest_open = float(latest["open"])
    latest_close = float(latest["close"])
    period_high_row = frame.loc[frame["high"].idxmax()]
    period_low_row = frame.loc[frame["low"].idxmin()]
    close_change = latest_close - float(first["close"])
    return (
        f"{timeframe}独立观察：本页使用 {format_dt(first['datetime'])} 到 {format_dt(latest['datetime'])} 的{timeframe}K线，"
        f"共 {len(frame)} 根。最新K线开盘 {latest_open:.2f}，高点 {latest_high:.2f}，"
        f"低点 {latest_low:.2f}，收盘 {latest_close:.2f}。"
        f"两个月内最高点 {float(period_high_row['high']):.2f}，时间 {format_dt(period_high_row['datetime'])}；"
        f"最低点 {float(period_low_row['low']):.2f}，时间 {format_dt(period_low_row['datetime'])}；"
        f"首尾收盘变化 {close_change:+.2f}。"
        f"{structure}"
        f"最大成交量K线是 {format_dt(max_volume_row['datetime'])}，成交量 {int(max_volume_row['volume'])}，"
        f"K线高点 {float(max_volume_row['high']):.2f}，低点 {float(max_volume_row['low']):.2f}，"
        f"收盘 {float(max_volume_row['close']):.2f}。{max_volume_story}"
        f"{oi_story}"
    )


def describe_price_structure(frame: pd.DataFrame) -> str:
    midpoint = max(1, len(frame) // 2)
    first_half = frame.iloc[:midpoint]
    second_half = frame.iloc[midpoint:]
    first_high = float(first_half["high"].max())
    first_low = float(first_half["low"].min())
    second_high = float(second_half["high"].max())
    second_low = float(second_half["low"].min())
    if second_high < first_high and second_low < first_low:
        return (
            f"高低点结构：前半段高点 {first_high:.2f}、低点 {first_low:.2f}，"
            f"后半段高点 {second_high:.2f}、低点 {second_low:.2f}，表现为高点下移、低点下移。"
        )
    if second_high > first_high and second_low > first_low:
        return (
            f"高低点结构：前半段高点 {first_high:.2f}、低点 {first_low:.2f}，"
            f"后半段高点 {second_high:.2f}、低点 {second_low:.2f}，表现为高点上移、低点上移。"
        )
    return (
        f"高低点结构：前半段高点 {first_high:.2f}、低点 {first_low:.2f}，"
        f"后半段高点 {second_high:.2f}、低点 {second_low:.2f}，表现为区间震荡。"
    )


def describe_max_volume_area(frame: pd.DataFrame, max_volume_row: pd.Series) -> str:
    after = frame[frame["datetime"] > max_volume_row["datetime"]]
    if after.empty:
        return "最大量能K线位于当前周期最后一根，后续行情尚未展开。"
    high = float(max_volume_row["high"])
    low = float(max_volume_row["low"])
    latest_close = float(frame.iloc[-1]["close"])
    inside_count = int(((after["close"] <= high) & (after["close"] >= low)).sum())
    if latest_close > high:
        location = "最新收盘在最大量能K线高点上方"
    elif latest_close < low:
        location = "最新收盘在最大量能K线低点下方"
    else:
        location = "最新收盘仍在最大量能K线高低点范围内"
    return (
        f"最大量能K线之后共有 {len(after)} 根K线，其中 {inside_count} 根收盘在这根K线高低点范围内，"
        f"{location}，说明后续行情主要围绕这根最大量能K线的高低点展开。"
    )


def describe_open_interest(frame: pd.DataFrame) -> str:
    first = frame.iloc[0]
    latest = frame.iloc[-1]
    oi_change = float(latest["open_interest"] - first["open_interest"])
    close_change = float(latest["close"] - first["close"])
    if close_change < 0 and oi_change > 0:
        attitude = "价格下行、持仓量增加，空方参与度增加。"
    elif close_change > 0 and oi_change > 0:
        attitude = "价格上行、持仓量增加，多方参与度增加。"
    elif close_change < 0 and oi_change < 0:
        attitude = "价格下行、持仓量减少，行情下跌过程中有持仓退出。"
    elif close_change > 0 and oi_change < 0:
        attitude = "价格上行、持仓量减少，行情上涨过程中有持仓退出。"
    else:
        attitude = "价格和持仓量首尾变化不明显。"
    return (
        f"持仓量从 {int(first['open_interest'])} 变化到 {int(latest['open_interest'])}，"
        f"变化 {oi_change:+.0f}；{attitude}"
    )


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
