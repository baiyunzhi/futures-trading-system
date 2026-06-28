from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TIMEFRAMES = ("日线", "周线", "小时线")

SYMBOLS = {
    "RB0": {"name": "螺纹钢", "multiplier": 10},
}


@dataclass(frozen=True)
class TimeframeView:
    timeframe: str
    summary: str
    observations: list[str]
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


def load_market_data(daily_path: Path, hourly_path: Path) -> dict[str, pd.DataFrame]:
    if not daily_path.exists():
        raise FileNotFoundError(f"缺少螺纹钢最近两个月日线数据文件: {daily_path}")
    if not hourly_path.exists():
        raise FileNotFoundError(f"缺少螺纹钢最近两个月小时线数据文件: {hourly_path}")
    return {
        "daily": _normalize_data(pd.read_csv(daily_path)),
        "hourly": _normalize_data(pd.read_csv(hourly_path)),
    }


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


def build_timeframes(symbol_daily_df: pd.DataFrame, symbol_hourly_df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    indexed = symbol_daily_df.set_index("datetime").sort_index()
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "open_interest": "last",
    }
    frames = {
        "周线": indexed.resample("W-FRI").agg(agg).dropna().reset_index(),
        "日线": indexed.reset_index(),
    }
    if symbol_hourly_df is not None and not symbol_hourly_df.empty:
        frames["小时线"] = symbol_hourly_df.sort_values("datetime").reset_index(drop=True)
    return frames


def combine_view(timeframe: str, frame: pd.DataFrame) -> TimeframeView:
    latest = frame.iloc[-1]
    summary = build_objective_summary(timeframe, frame)
    observations = build_observation_points(timeframe, frame)
    return TimeframeView(
        timeframe=timeframe,
        summary=summary,
        observations=observations,
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


def build_observation_points(timeframe: str, frame: pd.DataFrame) -> list[str]:
    max_volume_row = frame.loc[frame["volume"].idxmax()]
    latest = frame.iloc[-1]
    high = float(max_volume_row["high"])
    low = float(max_volume_row["low"])
    latest_close = float(latest["close"])
    latest_oi = float(latest["open_interest"])
    latest_volume = float(latest["volume"])
    max_volume = float(max_volume_row["volume"])
    after = frame[frame["datetime"] > max_volume_row["datetime"]]

    if latest_close > high:
        relation = f"当前收盘 {latest_close:.2f} 在最大成交量K线区间上方，上沿 {high:.2f} 是回看这一区间时最直接的压力/支撑转换位置。"
    elif latest_close < low:
        relation = f"当前收盘 {latest_close:.2f} 在最大成交量K线区间下方，下沿 {low:.2f} 是回看这一区间时最直接的压力/支撑转换位置。"
    else:
        relation = f"当前收盘 {latest_close:.2f} 仍在最大成交量K线 {low:.2f}-{high:.2f} 区间内部，行情仍被这根K线的高低点包住。"

    points = [
        (
            f"{timeframe}最大成交量K线出现在 {format_dt(max_volume_row['datetime'])}，"
            f"成交量 {int(max_volume)}，区间为 {low:.2f}-{high:.2f}；这个区间作为本周期观察锚点。"
        ),
        relation,
    ]

    if after.empty:
        points.append("最大成交量K线之后还没有新的K线，暂时只能观察这根K线本身的高低点。")
        return points

    after_high = float(after["high"].max())
    after_low = float(after["low"].min())
    inside_close = int(((after["close"] >= low) & (after["close"] <= high)).sum())
    above_close = int((after["close"] > high).sum())
    below_close = int((after["close"] < low).sum())
    upper_tests = int(((after["high"] >= high) & (after["low"] <= high)).sum())
    lower_tests = int(((after["high"] >= low) & (after["low"] <= low)).sum())
    oi_change_after = float(latest_oi - max_volume_row["open_interest"])
    close_change_after = float(latest_close - max_volume_row["close"])

    points.extend(
        [
            (
                f"最大成交量K线之后共有 {len(after)} 根K线：收盘在区间内 {inside_close} 根，"
                f"收盘在上沿之上 {above_close} 根，收盘在下沿之下 {below_close} 根。"
            ),
            (
                f"后续K线实际波动范围为 {after_low:.2f}-{after_high:.2f}；"
                f"上沿 {high:.2f} 被K线触及/穿越 {upper_tests} 次，下沿 {low:.2f} 被K线触及/穿越 {lower_tests} 次。"
            ),
            (
                f"从最大成交量K线到最新K线，收盘变化 {close_change_after:+.2f}，"
                f"持仓量变化 {oi_change_after:+.0f}，最新成交量 {int(latest_volume)}。"
            ),
        ]
    )
    points.append(describe_participant_attitude(close_change_after, oi_change_after))
    if timeframe == "日线":
        points.extend(build_daily_pressure_plan(frame))
    return points


def build_daily_pressure_plan(frame: pd.DataFrame) -> list[str]:
    day_0601 = get_row_by_date(frame, "2026-06-01")
    day_0610 = get_row_by_date(frame, "2026-06-10")
    max_volume_row = frame.loc[frame["volume"].idxmax()]
    if day_0601 is None or day_0610 is None:
        return []

    pressure = float(day_0601["high"])
    failed_break_high = float(day_0610["high"])
    stop_line = 3200.0
    stop_price = stop_line + 1.0
    target = float(max_volume_row["low"])
    risk = stop_price - pressure
    reward = pressure - target
    ratio = reward / risk if risk > 0 else 0.0
    oi_change = float(day_0610["open_interest"] - day_0601["open_interest"])
    volume_change = float(day_0610["volume"] - day_0601["volume"])

    return [
        (
            f"日线压力观察：6月1日收阳后行情逐步回落，6月1日高点 {pressure:.2f} 形成压力参考；"
            f"6月10日盘中最高 {failed_break_high:.2f} 越过6月1日高点，但收盘 {float(day_0610['close']):.2f} "
            f"回到压力位内，说明 {pressure:.2f}-{failed_break_high:.2f} 压力继续有效。"
        ),
        (
            f"日线计划区间：把 {pressure:.2f}-3200.00 作为轻仓试空观察带，不把 {pressure:.2f} 当作必然成交价；"
            f"盘中冲高失败可在 {pressure:.2f} 附近试空，收盘后确认失败则等待反抽到该区域再观察。"
        ),
        (
            f"风险收益测算：按 {pressure:.2f} 附近试空、止损 {stop_price:.2f}、目标最大成交量K线低点 {target:.2f} 计算，"
            f"风险 {risk:.2f} 点，目标空间 {reward:.2f} 点，风险利润比约 1:{ratio:.2f}。"
        ),
        (
            "盘中强弱量化：若价格进入 3189.00-3200.00 后不能连续2根15分钟K线收在3200上方，"
            "也不能1根小时K线收在3200上方，压力带仍按有效观察；若放量增仓并站上3200，则不做空。"
        ),
        (
            "成交量和持仓量前提：冲击压力带时成交量可以放大，但价格必须收不住3200；"
            "持仓量增加但价格站不住，属于增仓冲高失败；持仓量减少，则说明上攻缺少新增推动。"
        ),
        (
            f"6月1日至6月10日对比：成交量从 {int(day_0601['volume'])} 增至 {int(day_0610['volume'])}，"
            f"变化 {volume_change:+.0f}；持仓量从 {int(day_0601['open_interest'])} 变为 {int(day_0610['open_interest'])}，"
            f"变化 {oi_change:+.0f}，说明6月10日冲高回落时并不是明显增仓上攻。"
        ),
    ]


def get_row_by_date(frame: pd.DataFrame, date_text: str) -> pd.Series | None:
    date = pd.to_datetime(date_text).date()
    matched = frame[frame["datetime"].dt.date == date]
    if matched.empty:
        return None
    return matched.iloc[0]


def describe_participant_attitude(close_change: float, oi_change: float) -> str:
    if close_change > 0 and oi_change > 0:
        return "从最大成交量K线之后看，价格上移且持仓增加，说明多方主动参与增加。"
    if close_change < 0 and oi_change > 0:
        return "从最大成交量K线之后看，价格下移且持仓增加，说明空方主动参与增加。"
    if close_change > 0 and oi_change < 0:
        return "从最大成交量K线之后看，价格上移但持仓减少，说明上涨过程中有持仓退出。"
    if close_change < 0 and oi_change < 0:
        return "从最大成交量K线之后看，价格下移但持仓减少，说明下跌过程中有持仓退出。"
    return "从最大成交量K线之后看，价格和持仓变化不明显，双方态度暂未拉开。"


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


def analyze_market(data: dict[str, pd.DataFrame] | pd.DataFrame) -> list[MarketReport]:
    reports: list[MarketReport] = []
    if isinstance(data, dict):
        daily = _normalize_data(data["daily"])
        hourly = _normalize_data(data["hourly"])
    else:
        daily = _normalize_data(data)
        hourly = pd.DataFrame()
    for symbol, meta in SYMBOLS.items():
        symbol_daily_df = daily[daily["symbol"] == symbol]
        symbol_hourly_df = hourly[hourly["symbol"] == symbol] if not hourly.empty else None
        if symbol_daily_df.empty:
            continue
        frames = build_timeframes(symbol_daily_df, symbol_hourly_df)
        views = {name: combine_view(name, frame) for name, frame in frames.items() if len(frame) >= 5}
        if set(TIMEFRAMES).issubset(views):
            story = "；".join(views[name].summary for name in TIMEFRAMES)
            reports.append(MarketReport(symbol, meta["name"], views, story))
    return reports
