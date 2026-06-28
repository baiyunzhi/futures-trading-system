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
    workflow_alerts: list[str]
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
    workflow_alerts = build_workflow_alerts(timeframe, frame)
    return TimeframeView(
        timeframe=timeframe,
        summary=summary,
        observations=observations,
        workflow_alerts=workflow_alerts,
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
    return points


def build_workflow_alerts(timeframe: str, frame: pd.DataFrame) -> list[str]:
    return build_max_volume_anchor_workflow(timeframe, frame)


def build_max_volume_anchor_workflow(timeframe: str, frame: pd.DataFrame) -> list[str]:
    max_volume_row = frame.loc[frame["volume"].idxmax()]
    after = frame[frame["datetime"] > max_volume_row["datetime"]]
    high = float(max_volume_row["high"])
    low = float(max_volume_row["low"])
    close = float(max_volume_row["close"])
    volume = float(max_volume_row["volume"])
    latest = frame.iloc[-1]
    latest_close = float(latest["close"])
    latest_high = float(latest["high"])
    latest_low = float(latest["low"])

    alerts = [
        (
            f"{timeframe}工作流锚点：系统优先选择最大成交量K线作为观察点，时间 {format_dt(max_volume_row['datetime'])}，"
            f"成交量 {int(volume)}，区间 {low:.2f}-{high:.2f}，收盘 {close:.2f}。"
        ),
        (
            "工作流原则：最大成交量K线高点先作为压力观察，低点先作为支撑观察；"
            "后续行情突破、跌破、假突破、反抽失败，都围绕这根K线的高低点描述。"
        ),
    ]

    if after.empty:
        alerts.append("当前状态：最大成交量K线之后还没有后续K线，系统只能标出锚点，等待后续行情演化。")
        return alerts

    first_up_break = first_row(after[after["high"] > high])
    first_down_break = first_row(after[after["low"] < low])
    latest_up_reclaim = first_row(after[(after["close"] > high)])
    latest_down_accept = first_row(after[(after["close"] < low)])
    false_up = first_row(after[(after["high"] > high) & (after["close"] <= high)])
    false_down = first_row(after[(after["low"] < low) & (after["close"] >= low)])
    after_down_break = after
    if first_down_break is not None:
        after_down_break = after[after["datetime"] > first_down_break["datetime"]]
    retest_pressure = first_row(after_down_break[(after_down_break["high"] >= low) & (after_down_break["close"] < low)])
    retest_support = first_row(after[(after["low"] <= high) & (after["close"] > high)])

    inside_count = int(((after["close"] >= low) & (after["close"] <= high)).sum())
    above_count = int((after["close"] > high).sum())
    below_count = int((after["close"] < low).sum())
    alerts.append(
        f"后续演化统计：最大量K线之后共 {len(after)} 根K线，收盘在区间内 {inside_count} 根，"
        f"收盘在高点上方 {above_count} 根，收盘在低点下方 {below_count} 根。"
    )

    if latest_close > high:
        alerts.append(
            f"当前状态：最新收盘 {latest_close:.2f} 在最大量K线高点 {high:.2f} 上方，"
            "说明行情已经重新站到锚点区间上方，原高点压力被市场接受，不能继续按该高点压力做空。"
        )
    elif latest_close < low:
        alerts.append(
            f"当前状态：最新收盘 {latest_close:.2f} 在最大量K线低点 {low:.2f} 下方，"
            f"说明原支撑 {low:.2f} 已被跌破，系统把 {low:.2f} 转为反抽压力观察位。"
        )
    else:
        alerts.append(
            f"当前状态：最新收盘 {latest_close:.2f} 仍在最大量K线区间 {low:.2f}-{high:.2f} 内，"
            "行情仍围绕锚点区间震荡，等待靠近高点或低点后的量仓确认。"
        )

    if first_up_break is not None:
        alerts.append(
            f"高点演化：{format_dt(first_up_break['datetime'])} 盘中突破最大量K线高点 {high:.2f}。"
        )
    if false_up is not None:
        alerts.append(
            f"假突破提示：{format_dt(false_up['datetime'])} 盘中过高点但收盘 {float(false_up['close']):.2f} 未站上 {high:.2f}，"
            "高点压力仍有效，可进入轻仓试空观察。"
        )
    elif latest_up_reclaim is not None:
        alerts.append(
            f"继续突破提示：{format_dt(latest_up_reclaim['datetime'])} 收盘站上高点 {high:.2f}，"
            "说明按高点压力做空的判断失效，等待新的支撑/压力。"
        )

    if first_down_break is not None:
        alerts.append(
            f"低点演化：{format_dt(first_down_break['datetime'])} 盘中跌破最大量K线低点 {low:.2f}。"
        )
    if false_down is not None:
        alerts.append(
            f"跌破收回提示：{format_dt(false_down['datetime'])} 盘中跌破低点但收盘 {float(false_down['close']):.2f} 收回 {low:.2f} 上方，"
            "低点支撑暂时没有完全失效。"
        )
    elif latest_down_accept is not None:
        alerts.append(
            f"支撑转压力提示：{format_dt(latest_down_accept['datetime'])} 收盘跌破低点 {low:.2f}，"
            f"系统把 {low:.2f} 从支撑位改为反抽压力位。"
        )

    if retest_pressure is not None:
        alerts.append(
            f"反抽失败提示：{format_dt(retest_pressure['datetime'])} 反抽触及/越过 {low:.2f} 后收盘仍在其下，"
            f"说明 {low:.2f} 压力有效，后续可观察轻仓试空条件。"
        )
    elif retest_support is not None:
        alerts.append(
            f"回踩确认提示：{format_dt(retest_support['datetime'])} 回踩最大量K线高点附近后收盘仍在其上，"
            f"说明 {high:.2f} 可能从压力转为支撑。"
        )

    alerts.extend(build_dynamic_pressure_workflow(timeframe, after, low))
    alerts.extend(build_anchor_trade_plan(timeframe, max_volume_row, after, latest))
    return alerts


def build_dynamic_pressure_workflow(timeframe: str, after: pd.DataFrame, anchor_low: float) -> list[str]:
    if after.empty:
        return []

    below_anchor = after[after["close"] < anchor_low]
    if below_anchor.empty:
        return [
            f"动态压力工作流：价格还没有收盘跌破最大量K线低点 {anchor_low:.2f}，系统暂不下移压力位。"
        ]

    first_accept = below_anchor.iloc[0]
    before_next_break = after[after["datetime"] > first_accept["datetime"]]
    if before_next_break.empty:
        return [
            f"动态压力工作流：{format_dt(first_accept['datetime'])} 收盘跌破 {anchor_low:.2f} 后，"
            f"{anchor_low:.2f} 是当前第一压力位，等待反抽确认。"
        ]

    alerts = [
        (
            f"动态压力工作流：{format_dt(first_accept['datetime'])} 收盘跌破最大量K线低点 {anchor_low:.2f} 后，"
            f"{anchor_low:.2f} 从支撑位变成第一压力位。"
        )
    ]

    anchor_retest = first_row(
        before_next_break[
            (before_next_break["high"] >= anchor_low - 2)
            & (before_next_break["close"] <= anchor_low + 2)
        ]
    )
    if anchor_retest is not None:
        volume_change = float(anchor_retest["volume"] - first_accept["volume"])
        oi_change = float(anchor_retest["open_interest"] - first_accept["open_interest"])
        body = abs(float(anchor_retest["close"]) - float(anchor_retest["open"]))
        full_range = float(anchor_retest["high"] - anchor_retest["low"])
        alerts.append(
            f"反抽确认：{format_dt(anchor_retest['datetime'])} 反抽到 {anchor_low:.2f} 附近后停滞，"
            f"成交量较跌破时变化 {volume_change:+.0f}，持仓量变化 {oi_change:+.0f}，"
            f"K线实体 {body:.2f}、全长 {full_range:.2f}，说明反抽力度不足，第一压力位继续有效。"
        )

    phase_scope = before_next_break
    if anchor_retest is not None:
        phase_scope = before_next_break[before_next_break["datetime"] < anchor_retest["datetime"]]
    if phase_scope.empty:
        phase_scope = before_next_break
    first_low_row = phase_scope.loc[phase_scope["low"].idxmin()]
    phase_low = float(first_low_row["low"])
    phase_low_time = first_low_row["datetime"]
    after_phase_low = before_next_break[before_next_break["datetime"] > phase_low_time]
    phase_break = first_row(after_phase_low[after_phase_low["close"] < phase_low])

    alerts.append(
        f"阶段低点识别：跌破 {anchor_low:.2f} 后，系统识别到阶段低点 {phase_low:.2f}，"
        f"时间 {format_dt(phase_low_time)}。"
    )

    if phase_break is not None:
        alerts.append(
            f"压力位下移：{format_dt(phase_break['datetime'])} 收盘 {float(phase_break['close']):.2f} "
            f"跌破阶段低点 {phase_low:.2f}，系统把 {phase_low:.2f} 升级为新的压力位。"
        )
    else:
        alerts.append(
            f"等待提示：阶段低点 {phase_low:.2f} 尚未被收盘跌破，系统不下移压力位，继续等待。"
        )

    if timeframe == "小时线":
        alerts.extend(describe_latest_hourly_decline(after))
    return alerts


def describe_latest_hourly_decline(frame: pd.DataFrame, window: int = 20) -> list[str]:
    if len(frame) < window * 2:
        return []
    recent = frame.tail(window)
    previous = frame.iloc[-window * 2 : -window]
    recent_close_change = float(recent.iloc[-1]["close"] - recent.iloc[0]["close"])
    recent_oi_change = float(recent.iloc[-1]["open_interest"] - recent.iloc[0]["open_interest"])
    recent_volume = float(recent["volume"].sum())
    previous_volume = float(previous["volume"].sum())
    volume_change_pct = (recent_volume - previous_volume) / previous_volume * 100 if previous_volume > 0 else 0.0
    return [
        (
            f"最后下跌段量仓：最近 {window} 根小时K线价格从 {float(recent.iloc[0]['close']):.2f} "
            f"到 {float(recent.iloc[-1]['close']):.2f}，变化 {recent_close_change:+.2f}；"
            f"持仓量增加 {recent_oi_change:+.0f}，成交量合计较前 {window} 根变化 {volume_change_pct:+.1f}%。"
        ),
        (
            "量仓含义：价格继续下移时持仓量明显放大，但成交量没有同步成倍放大，"
            "说明空方参与仍在增加，行情不是单纯靠成交量爆发推动，而是持仓结构继续向下推进。"
        ),
    ]


def build_anchor_trade_plan(
    timeframe: str,
    max_volume_row: pd.Series,
    after: pd.DataFrame,
    latest: pd.Series,
) -> list[str]:
    high = float(max_volume_row["high"])
    low = float(max_volume_row["low"])
    latest_close = float(latest["close"])
    max_after_low = float(after["low"].min()) if not after.empty else float(latest["low"])
    max_after_high = float(after["high"].max()) if not after.empty else float(latest["high"])

    if latest_close < low:
        entry = current_pressure_level(after, low)
        stop = round_up_to_ten(entry) + 1.0
        target = max_after_low
        risk = stop - entry
        reward = entry - target
        ratio = reward / risk if risk > 0 else 0.0
        if reward <= 0 or ratio < 3:
            return [
                (
                    f"计划状态：价格已经离开当前压力位 {entry:.2f} 下方，但从反抽压力 {entry:.2f} 到当前可见低点 {target:.2f} "
                    f"的空间不足以形成1:3，系统提示等待，不追空。"
                )
            ]
        return [
            (
                f"计划提示：若后续反抽当前压力位 {entry:.2f} 附近不能重新站回，按支撑转压力观察；"
                f"参考止损 {stop:.2f}，目标看后续低点 {target:.2f}，风险利润比约 1:{ratio:.2f}。"
            )
        ]

    if latest_close > high:
        return [
            (
                f"计划状态：价格已经站上最大量K线高点 {high:.2f}，原压力失效；"
                "如果错过突破行情，系统不追多，等待回踩形成新的支撑/压力。"
            )
        ]

    return [
        (
            f"计划状态：价格仍在最大量K线区间 {low:.2f}-{high:.2f} 内，"
            "没有清晰支撑转压力或压力转支撑，系统提示等待。"
        )
    ]


def current_pressure_level(after: pd.DataFrame, anchor_low: float) -> float:
    below_anchor = after[after["close"] < anchor_low]
    if below_anchor.empty:
        return anchor_low
    first_accept = below_anchor.iloc[0]
    later = after[after["datetime"] > first_accept["datetime"]]
    if later.empty:
        return anchor_low
    anchor_retest = first_row(
        later[
            (later["high"] >= anchor_low - 2)
            & (later["close"] <= anchor_low + 2)
        ]
    )
    phase_scope = later
    if anchor_retest is not None:
        phase_scope = later[later["datetime"] < anchor_retest["datetime"]]
    if phase_scope.empty:
        phase_scope = later
    phase_low_row = phase_scope.loc[phase_scope["low"].idxmin()]
    phase_low = float(phase_low_row["low"])
    after_phase_low = later[later["datetime"] > phase_low_row["datetime"]]
    phase_break = first_row(after_phase_low[after_phase_low["close"] < phase_low])
    return phase_low if phase_break is not None else anchor_low


def first_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.iloc[0]

def round_up_to_ten(value: float) -> float:
    return float(int((value + 9) // 10 * 10))


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
