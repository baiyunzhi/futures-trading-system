from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SOURCE_ROOT = REPO_ROOT / "four_dimension_system"
DOCS_ROOT = REPO_ROOT / "docs" / "四维分析系统初版"

sys.path.insert(0, str(SOURCE_ROOT))

from market_system import (  # noqa: E402
    SYMBOLS,
    TIMEFRAMES,
    build_timeframes,
    current_pressure_level,
    format_dt,
    load_market_data,
    readiness_from_tags,
    state_tags,
    waiting_reason_from_state,
)


DATA_FILES = {
    "RB0": {
        "daily": SOURCE_ROOT / "rb_recent_daily.csv",
        "hourly": SOURCE_ROOT / "rb_recent_hourly.csv",
    },
    "V0": {
        "daily": SOURCE_ROOT / "v_recent_daily.csv",
        "hourly": SOURCE_ROOT / "v_recent_hourly.csv",
    },
}


def pct(distance: float, close: float) -> float:
    return abs(distance - close) / close * 100 if close else 999.0


def anchor_event_profile(data: pd.DataFrame, anchor: pd.Series) -> dict[str, object]:
    ordered = data.sort_values("datetime").reset_index(drop=True)
    anchor_time = anchor["datetime"]
    idx_matches = ordered.index[ordered["datetime"] == anchor_time].tolist()
    idx = idx_matches[0] if idx_matches else int(ordered["volume"].idxmax())
    previous = ordered.iloc[idx - 1] if idx > 0 else None
    high = float(anchor["high"])
    low = float(anchor["low"])
    open_ = float(anchor["open"])
    close = float(anchor["close"])
    volume = float(anchor["volume"])
    ranked_volume = ordered["volume"].sort_values(ascending=False).reset_index(drop=True)
    second_volume = float(ranked_volume.iloc[1]) if len(ranked_volume) > 1 else volume
    volume_ratio = volume / second_volume if second_volume else 1.0
    price_range = max(0.01, high - low)
    body = abs(close - open_)
    upper_shadow = high - max(open_, close)
    lower_shadow = min(open_, close) - low
    direction = "收阳" if close >= open_ else "收阴"
    if body <= price_range * 0.25:
        shape = "小实体，说明多空在该区间激烈换手后暂未分出明显胜负"
    elif upper_shadow >= price_range * 0.4:
        shape = "长上影，说明上方抛压或空方反击明显"
    elif lower_shadow >= price_range * 0.4:
        shape = "长下影，说明下方承接或多方反击明显"
    elif close > open_:
        shape = "实体收阳，说明该周期多方推进更主动"
    else:
        shape = "实体收阴，说明该周期空方推进更主动"
    if previous is None:
        oi_text = "持仓量没有前一根K线可比。"
        oi_tag = "锚点持仓待比对"
    else:
        oi_delta = float(anchor["open_interest"] - previous["open_interest"])
        if oi_delta > 0:
            oi_text = f"持仓较前一根增加 {oi_delta:.0f}，说明这根放量包含新资金参与。"
            oi_tag = "锚点增仓"
        elif oi_delta < 0:
            oi_text = f"持仓较前一根减少 {abs(oi_delta):.0f}，说明这根放量包含持仓退出。"
            oi_tag = "锚点减仓"
        else:
            oi_text = "持仓较前一根基本不变，说明主要是换手。"
            oi_tag = "锚点换手"
    event_tag = "异常放量事件" if volume_ratio >= 1.2 else "周期最大量事件"
    text = (
        f"最大量K线性质：{direction}，{shape}；成交量为本周期最大，"
        f"是次大量的 {volume_ratio:.2f} 倍。{oi_text}"
    )
    return {"tags": [event_tag, oi_tag], "text": text}


def location_volume_confirmation(
    data: pd.DataFrame,
    pressure: float,
    support: float,
    tags: list[str],
) -> dict[str, object]:
    ordered = data.sort_values("datetime").reset_index(drop=True)
    if len(ordered) < 2:
        return {"tags": ["量仓确认不足"], "text": "量仓确认：K线数量不足，不能判断观察位附近的态度。"}
    latest = ordered.iloc[-1]
    previous = ordered.iloc[-2]
    latest_close = float(latest["close"])
    latest_high = float(latest["high"])
    latest_low = float(latest["low"])
    latest_volume = float(latest["volume"])
    previous_volume = float(previous["volume"])
    oi_delta = float(latest["open_interest"] - previous["open_interest"])
    volume_text = "成交量放大" if latest_volume > previous_volume else "成交量缩小" if latest_volume < previous_volume else "成交量持平"
    oi_text = "持仓增加" if oi_delta > 0 else "持仓减少" if oi_delta < 0 else "持仓持平"
    near_pressure = pct(pressure, latest_close) <= 2.5 or abs(latest_high - pressure) / latest_close * 100 <= 1.2
    near_support = pct(support, latest_close) <= 2.5 or abs(latest_low - support) / latest_close * 100 <= 1.2

    if ("动态观察位下移" in tags or "支撑转压力" in tags) and near_pressure:
        if latest_close <= pressure and latest_volume <= previous_volume and oi_delta <= 0:
            tag = "压力位反抽无力"
            result = "反抽没有得到成交量和持仓量支持，压力位继续有效。"
        elif latest_close <= pressure and oi_delta > 0:
            tag = "压力位空方防守"
            result = "反抽未站回压力位且持仓增加，空方仍在压力位附近参与。"
        elif latest_close > pressure and latest_volume > previous_volume and oi_delta > 0:
            tag = "压力位被挑战"
            result = "价格站回压力位且成交量、持仓量同步增加，原压力位需要重新验证。"
        else:
            tag = "压力位待确认"
            result = "价格接近压力位，但成交量和持仓量没有给出单边态度。"
        text = f"量仓确认：当前在压力观察位附近，{volume_text}、{oi_text}。{result}"
        return {"tags": [tag], "text": text}

    if "压力转支撑" in tags and near_support:
        if latest_close >= support and latest_volume <= previous_volume and oi_delta <= 0:
            tag = "支撑位回踩无力"
            result = "回踩没有得到下行动能支持，支撑位继续有效。"
        elif latest_close >= support and oi_delta > 0:
            tag = "支撑位多方防守"
            result = "价格守住支撑且持仓增加，多方仍在支撑位附近参与。"
        elif latest_close < support and latest_volume > previous_volume and oi_delta > 0:
            tag = "支撑位被挑战"
            result = "价格跌回支撑位下方且成交量、持仓量同步增加，原支撑位需要重新验证。"
        else:
            tag = "支撑位待确认"
            result = "价格接近支撑位，但成交量和持仓量没有给出单边态度。"
        text = f"量仓确认：当前在支撑观察位附近，{volume_text}、{oi_text}。{result}"
        return {"tags": [tag], "text": text}

    text = (
        f"量仓确认：当前未贴近最近观察位；最新一根{volume_text}、{oi_text}，"
        "只能说明当下态度，不能作为支撑/压力确认。"
    )
    return {"tags": ["远离观察位待确认"], "text": text}


def build_symbol_priorities(frames_by_symbol: dict[str, dict[str, pd.DataFrame]]) -> dict[str, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol, frames in frames_by_symbol.items():
        daily = frames.get("日线")
        if daily is None or daily.empty:
            continue
        data = daily.sort_values("datetime").reset_index(drop=True)
        anchor = data.loc[data["volume"].idxmax()]
        latest = data.iloc[-1]
        first = data.iloc[0]
        after = data[data["datetime"] > anchor["datetime"]]
        latest_close = float(latest["close"])
        first_close = float(first["close"])
        anchor_low = float(anchor["low"])
        close_change_pct = (latest_close - first_close) / first_close * 100 if first_close else 0.0
        oi_change = float(latest["open_interest"] - first["open_interest"])
        below_ratio = float((after["close"] < anchor_low).sum() / len(after)) if len(after) else 0.0
        chain = build_observation_chain(data, anchor)
        chain_count = len(chain.get("levels", []))
        latest_near_low = float(latest["low"]) <= float(data["low"].min()) * 1.01
        weakness_score = (
            max(0.0, -close_change_pct) * 4.0
            + chain_count * 18.0
            + below_ratio * 30.0
            + (18.0 if latest_close < anchor_low else 0.0)
            + (14.0 if latest_near_low else 0.0)
            + (10.0 if close_change_pct < 0 and oi_change > 0 else 0.0)
        )
        rows.append(
            {
                "symbol": symbol,
                "name": SYMBOLS[symbol]["name"],
                "score": weakness_score,
                "close_change_pct": close_change_pct,
                "oi_change": oi_change,
                "chain_count": chain_count,
                "latest_close": latest_close,
            }
        )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    priorities: dict[str, dict[str, object]] = {}
    for rank, row in enumerate(rows, start=1):
        priority = "A类优先" if rank == 1 and float(row["score"]) >= 45 else "B类备选" if float(row["score"]) >= 25 else "C类等待"
        row["rank"] = rank
        row["priority"] = priority
        row["summary"] = (
            f"日线两个月收盘变化 {float(row['close_change_pct']):.2f}%，"
            f"动态观察位 {int(row['chain_count'])} 层，"
            f"持仓变化 {float(row['oi_change']):.0f}。"
        )
        priorities[str(row["symbol"])] = row
    return priorities


def period_card(
    symbol: str,
    name: str,
    timeframe: str,
    frame: pd.DataFrame,
    priority: dict[str, object] | None,
) -> str:
    data = frame.sort_values("datetime").reset_index(drop=True)
    anchor = data.loc[data["volume"].idxmax()]
    latest = data.iloc[-1]
    first = data.iloc[0]
    after = data[data["datetime"] > anchor["datetime"]]

    anchor_high = float(anchor["high"])
    anchor_low = float(anchor["low"])
    latest_close = float(latest["close"])
    close_change = latest_close - float(first["close"])
    oi_change = float(latest["open_interest"] - first["open_interest"])
    observation_chain = build_observation_chain(data, anchor)

    if observation_chain["current_pressure"] is not None:
        pressure = float(observation_chain["current_pressure"])
        support = float(observation_chain["current_support"])
    elif latest_close < anchor_low:
        pressure = current_pressure_level(after, anchor_low)
        support = float(after["low"].min()) if not after.empty else float(latest["low"])
    elif latest_close > anchor_high:
        pressure = float(after["high"].max()) if not after.empty else float(latest["high"])
        support = anchor_high
    else:
        pressure = anchor_high
        support = anchor_low

    pressure_distance = pct(pressure, latest_close)
    support_distance = pct(support, latest_close)
    anchor_state = analyze_anchor_state(data, anchor, after, latest_close, pressure, support)
    anchor_event = anchor_event_profile(data, anchor)
    below_ratio = float((after["close"] < anchor_low).sum() / len(after)) if len(after) else 0.0
    above_ratio = float((after["close"] > anchor_high).sum() / len(after)) if len(after) else 0.0
    inside_ratio = float(((after["close"] >= anchor_low) & (after["close"] <= anchor_high)).sum() / len(after)) if len(after) else 0.0

    tags = state_tags(
        latest_close=latest_close,
        anchor_high=anchor_high,
        anchor_low=anchor_low,
        close_change=close_change,
        oi_change=oi_change,
        below_ratio=below_ratio,
        above_ratio=above_ratio,
        inside_ratio=inside_ratio,
        pressure_distance=pressure_distance,
        support_distance=support_distance,
    )
    tags = normalize_distance_tags(tags, observation_chain, pressure_distance, support_distance)
    tags.extend(chain_tags(observation_chain))
    tags.append(anchor_state["state"])
    tags.extend(anchor_event["tags"])
    volume_confirmation = location_volume_confirmation(data, pressure, support, tags)
    tags.extend(volume_confirmation["tags"])
    weak_observation = weak_consolidation_observation(data, observation_chain)
    tags.extend(weak_observation["tags"])
    if priority:
        tags.append(str(priority["priority"]))
    readiness_level, readiness_text = readiness_from_current_observation(tags, pressure_distance, support_distance)
    wait_reason = waiting_reason_from_current_observation(tags, pressure_distance, support_distance, readiness_level)
    wait_reason = combine_wait_reason(wait_reason, anchor_state)
    wait_reason = combine_volume_confirmation(wait_reason, volume_confirmation)
    wait_reason = combine_weak_wait_reason(wait_reason, weak_observation)
    status = build_status(
        tags,
        close_change,
        oi_change,
        below_ratio,
        above_ratio,
        inside_ratio,
        pressure,
        support,
        pressure_distance,
        support_distance,
        observation_chain,
    )
    observe = (
        f"压力 {pressure:.2f}，距最新收盘 {pressure_distance:.2f}%；"
        f"支撑 {support:.2f}，距最新收盘 {support_distance:.2f}%。"
        f"最大量K线 {format_dt(anchor['datetime'])}，区间 {anchor_low:.2f}-{anchor_high:.2f}。"
        f"{anchor_event['text']}"
        f"{observation_chain['text']}"
        f"{anchor_state['observe']}"
    )
    invalidation = build_invalidation(tags, pressure, support, anchor_high, anchor_low, anchor_state)
    prepare_text = build_prepare_text(readiness_text, priority, weak_observation)
    tag_html = "".join(f'<span class="tag">{escape(tag)}</span>' for tag in tags)
    chart_id = f"{symbol}-{timeframe}"
    return f"""
      <article class="period-card level-{readiness_level}">
        <div class="card-head">
          <h3>{escape(name)}({escape(symbol)}) · {escape(timeframe)}</h3>
          <span>{escape(readiness_text)}</span>
        </div>
        <div class="period-layout">
          <div class="chart-pane">
            {kline_svg(data, chart_id, anchor_state, observation_chain)}
            <div class="kline-detail" id="detail-{escape(chart_id)}">点击K线查看该根K线的时间、开高低收、成交量、持仓量。</div>
          </div>
          <aside class="info-pane">
            <div class="tag-row">{tag_html}</div>
            {line_item("1. 当前状态", status)}
            {line_item("2. 当前观察位", observe)}
            {line_item("3. 等待原因", wait_reason)}
            {line_item("4. 准备等级", prepare_text)}
            {line_item("5. 失效条件", invalidation)}
          </aside>
        </div>
      </article>
    """


def build_status(
    tags: list[str],
    close_change: float,
    oi_change: float,
    below_ratio: float,
    above_ratio: float,
    inside_ratio: float,
    pressure: float,
    support: float,
    pressure_distance: float,
    support_distance: float,
    observation_chain: dict[str, object],
) -> str:
    if "动态观察位下移" in tags:
        control = f"空方控制，观察位已下移到 {pressure:.2f}"
    elif "支撑转压力" in tags:
        control = "空方控制"
    elif "压力转支撑" in tags:
        control = "多方控制"
    else:
        control = "区间拉锯"
    if close_change < 0 and oi_change > 0:
        attitude = "价格下移、持仓增加，空方主动参与增加。"
    elif close_change > 0 and oi_change > 0:
        attitude = "价格上移、持仓增加，多方主动参与增加。"
    elif close_change < 0 and oi_change < 0:
        attitude = "价格下移、持仓减少，下跌中有持仓退出。"
    elif close_change > 0 and oi_change < 0:
        attitude = "价格上移、持仓减少，上涨中有持仓退出。"
    else:
        attitude = "价格和持仓变化不明显，双方态度暂未拉开。"
    return (
        f"{control}。锚点后收在低点下方占比 {below_ratio * 100:.1f}%，"
        f"收在高点上方占比 {above_ratio * 100:.1f}%，"
        f"收在区间内占比 {inside_ratio * 100:.1f}%。"
        f"当前压力距离 {pressure_distance:.2f}%，当前支撑距离 {support_distance:.2f}%。{attitude}"
    )


def chain_tags(observation_chain: dict[str, object]) -> list[str]:
    if observation_chain["current_pressure"] is None:
        return []
    levels = observation_chain.get("levels", [])
    if len(levels) >= 2:
        return ["动态观察位下移"]
    return ["支撑转压力"]


def normalize_distance_tags(
    tags: list[str],
    observation_chain: dict[str, object],
    pressure_distance: float,
    support_distance: float,
) -> list[str]:
    out = [tag for tag in tags if tag not in ("接近观察位", "远离观察位")]
    if observation_chain["current_pressure"] is not None or "支撑转压力" in out:
        out.append("接近当前压力位" if pressure_distance <= 2.5 else "远离当前压力位")
    elif "压力转支撑" in out:
        out.append("接近当前支撑位" if support_distance <= 2.5 else "远离当前支撑位")
    elif min(pressure_distance, support_distance) <= 2:
        out.append("接近观察位")
    else:
        out.append("远离观察位")
    return out


def readiness_from_current_observation(
    tags: list[str],
    pressure_distance: float,
    support_distance: float,
) -> tuple[int, str]:
    if "动态观察位下移" in tags or "支撑转压力" in tags:
        if pressure_distance <= 1:
            return 3, "3级：位置明确，可制定计划"
        if pressure_distance <= 2.5:
            return 2, "2级：反抽确认中"
        if pressure_distance <= 5:
            return 1, "1级：接近观察位"
        return 0, "0级：等待"
    if "压力转支撑" in tags:
        if support_distance <= 1:
            return 3, "3级：位置明确，可制定计划"
        if support_distance <= 2.5:
            return 2, "2级：回踩确认中"
        if support_distance <= 5:
            return 1, "1级：接近观察位"
        return 0, "0级：等待"
    return readiness_from_tags(tags, pressure_distance, support_distance)


def waiting_reason_from_current_observation(
    tags: list[str],
    pressure_distance: float,
    support_distance: float,
    readiness_level: int,
) -> str:
    if "动态观察位下移" in tags or "支撑转压力" in tags:
        if readiness_level == 0:
            return f"当前价格距离最近压力观察位 {pressure_distance:.2f}%，不适合追空，等待反抽靠近当前压力位。"
        if readiness_level in (1, 2):
            return "价格已接近当前压力观察位，等待收盘不能重新站回，同时观察成交量和持仓量是否不支持上攻。"
        return "当前压力观察位已经明确，下一步只等待具体触发条件，不提前追价。"
    if "压力转支撑" in tags:
        if readiness_level == 0:
            return f"当前价格距离最近支撑观察位 {support_distance:.2f}%，不适合追多，等待回踩靠近当前支撑位。"
        if readiness_level in (1, 2):
            return "价格已接近当前支撑观察位，等待收盘不能跌回，同时观察成交量和持仓量是否支持。"
        return "当前支撑观察位已经明确，下一步只等待具体触发条件。"
    return waiting_reason_from_state(tags, pressure_distance, support_distance, readiness_level)


def build_observation_chain(data: pd.DataFrame, anchor: pd.Series) -> dict[str, object]:
    anchor_low = float(anchor["low"])
    latest = data.iloc[-1]
    latest_close = float(latest["close"])
    after = data[data["datetime"] > anchor["datetime"]].reset_index(drop=True)
    chain: list[dict[str, object]] = []
    if after.empty:
        return {
            "levels": chain,
            "current_pressure": None,
            "current_support": anchor_low,
            "text": "观察位链条：最大量K线之后暂无演变。"
        }

    if not after[after["close"] < anchor_low].empty:
        chain.append(
            {
                "level": anchor_low,
                "time": anchor["datetime"],
                "kind": "原锚点低点失守",
            }
        )

    previous_level = anchor_low
    lows = after.reset_index(drop=True)
    for idx in range(len(lows) - 3):
        row = lows.iloc[idx]
        level = float(row["low"])
        if level >= previous_level * 0.992:
            continue
        next_closes = lows.iloc[idx + 1 : idx + 4]["close"]
        if len(next_closes) < 2 or not bool((next_closes > level).all()):
            continue
        later = lows.iloc[idx + 4 :]
        if later.empty or later[later["close"] < level].empty:
            continue
        item = {
            "level": level,
            "time": row["datetime"],
            "kind": "阶段支撑失守",
            "idx": idx,
        }
        if chain and chain[-1].get("kind") == "阶段支撑失守" and idx - int(chain[-1].get("idx", -99)) <= 3:
            if level < float(chain[-1]["level"]):
                chain[-1] = item
        else:
            chain.append(item)
        previous_level = level

    # Keep the chain readable and focused on the latest valid observation levels.
    if len(chain) > 5:
        chain = [chain[0], *chain[-4:]]
    current_pressure = float(chain[-1]["level"]) if chain else None
    if current_pressure is not None:
        later_than_pressure = data[data["datetime"] > chain[-1]["time"]]
        support_scope = later_than_pressure if not later_than_pressure.empty else after
        current_support = float(support_scope["low"].min())
    else:
        current_support = float(after["low"].min()) if latest_close < anchor_low else anchor_low
    if chain:
        chain_text = " → ".join(f"{float(item['level']):.2f}" for item in chain)
        text = f"观察位链条：{chain_text}。当前按最近失守位 {current_pressure:.2f} 作为压力观察位。"
        if current_support < current_pressure:
            text += f" 当前下方未确认支撑为 {current_support:.2f}。"
    else:
        text = "观察位链条：价格尚未形成有效的支撑压力下移链条。"
    return {
        "levels": chain,
        "current_pressure": current_pressure,
        "current_support": current_support,
        "text": text,
    }


def analyze_anchor_state(
    data: pd.DataFrame,
    anchor: pd.Series,
    after: pd.DataFrame,
    latest_close: float,
    pressure: float,
    support: float,
) -> dict[str, object]:
    anchor_volume = float(anchor["volume"])
    anchor_high = float(anchor["high"])
    anchor_low = float(anchor["low"])
    old_distance = min(pct(pressure, latest_close), pct(support, latest_close))
    if after.empty:
        return {
            "state": "原锚点有效",
            "observe": "最大量K线之后还没有新K线，继续等待后续演化。",
            "wait": "暂无后续K线，不能重建锚点。",
            "candidate": None,
        }

    recent_start = max(0, len(data) - max(6, len(data) // 3))
    recent = data.iloc[recent_start:]
    recent_after = recent[recent["datetime"] > anchor["datetime"]]
    scope = recent_after if not recent_after.empty else after
    candidate = scope.loc[scope["volume"].idxmax()]
    candidate_volume = float(candidate["volume"])
    candidate_ratio = candidate_volume / anchor_volume if anchor_volume else 0.0
    candidate_low = float(candidate["low"])
    candidate_high = float(candidate["high"])
    post_anchor_low = float(after["low"].min())
    post_anchor_high = float(after["high"].max())
    near_new_low = candidate_low <= post_anchor_low * 1.01
    near_new_high = candidate_high >= post_anchor_high * 0.99
    close_near_candidate = candidate_low <= latest_close <= candidate_high
    candidate_valid = candidate_ratio >= 0.45 and (near_new_low or near_new_high or close_near_candidate)

    if candidate_valid:
        return {
            "state": "候选新锚点",
            "observe": (
                f"候选新锚点 {format_dt(candidate['datetime'])}，成交量为原锚点的 {candidate_ratio * 100:.1f}%，"
                f"区间 {candidate_low:.2f}-{candidate_high:.2f}。"
            ),
            "wait": "后续观察价格是否继续围绕候选新锚点高低点波动，确认后再替换原锚点。",
            "candidate": candidate,
        }

    if old_distance > 8:
        return {
            "state": "锚点待重建",
            "observe": (
                f"当前价格距离原锚点相关观察位 {old_distance:.2f}%，旧锚点参考距离偏远，"
                "等待新的放量K线形成观察区间。"
            ),
            "wait": "旧锚点距离过远且没有合格候选锚点，当前以等待新锚点为主。",
            "candidate": None,
        }

    return {
        "state": "原锚点有效",
        "observe": "当前仍以原最大成交量K线作为观察锚点。",
        "wait": "原锚点仍有效，继续观察其高低点的支撑压力转换。",
        "candidate": None,
    }


def combine_wait_reason(wait_reason: str, anchor_state: dict[str, object]) -> str:
    return f"{wait_reason}锚点提示：{anchor_state['wait']}"


def combine_volume_confirmation(wait_reason: str, volume_confirmation: dict[str, object]) -> str:
    return f"{wait_reason}{volume_confirmation['text']}"


def weak_consolidation_observation(data: pd.DataFrame, observation_chain: dict[str, object]) -> dict[str, object]:
    levels = observation_chain.get("levels", [])
    if not levels:
        return {"tags": [], "text": "", "rr_text": ""}
    pressure_time = levels[-1]["time"]
    pressure = float(levels[-1]["level"])
    scope = data[data["datetime"] >= pressure_time].reset_index(drop=True)
    if len(scope) < 5:
        return {"tags": [], "text": "", "rr_text": ""}

    latest = scope.iloc[-1]
    latest_low = float(latest["low"])
    candidates: list[dict[str, object]] = []
    for idx in range(0, len(scope) - 3):
        row = scope.iloc[idx]
        level = float(row["low"])
        if level >= pressure:
            continue
        following = scope.iloc[idx + 1 : -1]
        if len(following) < 2:
            continue
        if not bool((following["close"] > level).all()):
            continue
        if latest_low >= level:
            continue
        resistance_high = float(scope.iloc[idx:-1]["high"].max())
        entry = float(row["high"])
        stop_buffer = max(1.0, round(entry * 0.0003))
        stop = resistance_high + stop_buffer
        risk = max(0.01, stop - entry)
        first_reward = max(0.0, entry - level)
        extended_reward = max(0.0, entry - latest_low)
        candidates.append(
            {
                "time": row["datetime"],
                "level": level,
                "entry": entry,
                "resistance_high": resistance_high,
                "stop": stop,
                "first_rr": first_reward / risk if risk else 0.0,
                "extended_rr": extended_reward / risk if risk else 0.0,
                "latest_low": latest_low,
            }
        )
    if not candidates:
        return {"tags": [], "text": "", "rr_text": ""}

    item = candidates[-1]
    resistance_text = (
        f"{float(item['entry']):.2f}"
        if abs(float(item["entry"]) - float(item["resistance_high"])) < 0.01
        else f"{float(item['entry']):.2f}-{float(item['resistance_high']):.2f}"
    )
    rr_pass = float(item["first_rr"]) >= 3.0
    tags = ["弱势震荡", "风险收益合格" if rr_pass else "风险收益不足"]
    text = (
        f"弱势震荡观察：{format_dt(item['time'])} 前低 {float(item['level']):.2f} 被跌破前，"
        f"价格在 {resistance_text} 附近反抽停滞；再次跌破后，该前低成为新压力观察位。"
    )
    rr_text = (
        f"若只在反抽 {float(item['entry']):.2f} 附近试空，止损放在 {float(item['stop']):.2f} 上方，"
        f"第一目标 {float(item['level']):.2f}，风险收益比约 1:{float(item['first_rr']):.1f}；"
        f"扩展到最新低点 {float(item['latest_low']):.2f} 约 1:{float(item['extended_rr']):.1f}。"
        "当前位置若已远离反抽区，只等待，不追空。"
    )
    return {"tags": tags, "text": text, "rr_text": rr_text}


def combine_weak_wait_reason(wait_reason: str, weak_observation: dict[str, object]) -> str:
    text = str(weak_observation.get("text") or "")
    if not text:
        return wait_reason
    return f"{wait_reason}{text}"


def build_prepare_text(
    readiness_text: str,
    priority: dict[str, object] | None,
    weak_observation: dict[str, object],
) -> str:
    parts = [readiness_text]
    if priority:
        parts.append(f"品种优先级 {priority['priority']}，排名第 {priority['rank']}；{priority['summary']}")
    rr_text = str(weak_observation.get("rr_text") or "")
    if rr_text:
        parts.append(rr_text)
    return " ".join(parts)


def build_invalidation(
    tags: list[str],
    pressure: float,
    support: float,
    anchor_high: float,
    anchor_low: float,
    anchor_state: dict[str, object],
) -> str:
    if "动态观察位下移" in tags or "支撑转压力" in tags:
        return (
            f"若后续收盘重新站回当前压力观察位 {pressure:.2f} 上方，"
            "并且成交量、持仓量支持上攻，则当前压力位失效；系统回看上一层压力或等待新锚点。"
        )
    if anchor_state["state"] == "候选新锚点":
        candidate = anchor_state["candidate"]
        return (
            f"若后续价格不再围绕候选新锚点 {format_dt(candidate['datetime'])} 的高低点波动，"
            "候选锚点失效，继续使用原锚点或等待新锚点。"
        )
    if anchor_state["state"] == "锚点待重建":
        return "若后续出现新的放量K线并形成高低点争夺区，锚点待重建状态结束，切换到候选新锚点观察。"
    if "支撑转压力" in tags:
        return (
            f"若后续收盘重新站回压力位 {pressure:.2f} 上方，并且成交量、持仓量支持上攻，"
            "空方观察失效，等待新锚点。"
        )
    if "压力转支撑" in tags:
        return (
            f"若后续收盘跌回支撑位 {support:.2f} 下方，并且成交量、持仓量支持下行，"
            "多方观察失效，等待新锚点。"
        )
    return (
        f"若收盘有效离开最大量K线区间 {anchor_low:.2f}-{anchor_high:.2f}，"
        "当前拉锯状态失效，重新定义支撑压力。"
    )


def line_item(title: str, body: str) -> str:
    number, label = title.split(". ", 1) if ". " in title else ("", title)
    return f"""
      <section class="line-item">
        <b><span>{escape(number)}</span>{escape(label)}</b>
        <p>{escape(body)}</p>
      </section>
    """


def kline_svg(
    frame: pd.DataFrame,
    chart_id: str,
    anchor_state: dict[str, object] | None = None,
    observation_chain: dict[str, object] | None = None,
    width: int = 1200,
    height: int = 560,
) -> str:
    data = frame.sort_values("datetime").reset_index(drop=True)
    if data.empty:
        return ""
    top_pad = 24
    price_h = 300
    sub_top = 390
    sub_h = 110
    bottom_pad = 34
    left_pad = 54
    right_pad = 16
    plot_w = width - left_pad - right_pad
    price_high = float(data["high"].max())
    price_low = float(data["low"].min())
    price_span = price_high - price_low if price_high > price_low else 1.0
    high_row = data.loc[data["high"].idxmax()]
    low_row = data.loc[data["low"].idxmin()]
    max_volume_row = data.loc[data["volume"].idxmax()]
    max_volume_high = float(max_volume_row["high"])
    max_volume_low = float(max_volume_row["low"])
    max_volume = float(data["volume"].max()) if float(data["volume"].max()) > 0 else 1.0
    oi_high = float(data["open_interest"].max())
    oi_low = float(data["open_interest"].min())
    oi_span = oi_high - oi_low if oi_high > oi_low else 1.0
    step = plot_w / max(1, len(data))
    candle_w = max(6, min(12, step * 0.48))

    def y(price: float) -> float:
        return top_pad + (price_high - price) / price_span * price_h

    def volume_y(volume: float) -> float:
        return sub_top + sub_h - volume / max_volume * sub_h

    def oi_y(value: float) -> float:
        return sub_top + (oi_high - value) / oi_span * sub_h

    def x_at(row_index: int) -> float:
        return left_pad + row_index * step + step / 2

    elements = [
        f'<svg viewBox="0 0 {width} {height}" class="kline" role="img">',
        f'<line x1="{left_pad}" y1="{top_pad}" x2="{left_pad}" y2="{top_pad + price_h}" class="axis"/>',
        f'<line x1="{left_pad}" y1="{top_pad + price_h}" x2="{width - right_pad}" y2="{top_pad + price_h}" class="axis"/>',
        f'<text x="8" y="{top_pad + 4}" class="axis-label">{price_high:.2f}</text>',
        f'<text x="8" y="{top_pad + price_h}" class="axis-label">{price_low:.2f}</text>',
        f'<line x1="{left_pad}" y1="{y(max_volume_high):.1f}" x2="{width - right_pad}" y2="{y(max_volume_high):.1f}" class="max-volume-line"/>',
        f'<line x1="{left_pad}" y1="{y(max_volume_low):.1f}" x2="{width - right_pad}" y2="{y(max_volume_low):.1f}" class="max-volume-line"/>',
        f'<text x="{width - right_pad - 190}" y="{y(max_volume_high) - 6:.1f}" class="max-volume-label">最大量高 {max_volume_high:.2f}</text>',
        f'<text x="{width - right_pad - 190}" y="{y(max_volume_low) + 16:.1f}" class="max-volume-label">最大量低 {max_volume_low:.2f}</text>',
        f'<text x="8" y="{sub_top + 12}" class="axis-label">成交量/持仓量</text>',
        f'<line x1="{left_pad}" y1="{sub_top + sub_h}" x2="{width - right_pad}" y2="{sub_top + sub_h}" class="axis"/>',
    ]
    if anchor_state and anchor_state.get("candidate") is not None:
        candidate = anchor_state["candidate"]
        candidate_high = float(candidate["high"])
        candidate_low = float(candidate["low"])
        elements.extend(
            [
                f'<line x1="{left_pad}" y1="{y(candidate_high):.1f}" x2="{width - right_pad}" y2="{y(candidate_high):.1f}" class="candidate-line"/>',
                f'<line x1="{left_pad}" y1="{y(candidate_low):.1f}" x2="{width - right_pad}" y2="{y(candidate_low):.1f}" class="candidate-line"/>',
                f'<text x="{width - right_pad - 190}" y="{y(candidate_high) - 6:.1f}" class="candidate-label">候选高 {candidate_high:.2f}</text>',
                f'<text x="{width - right_pad - 190}" y="{y(candidate_low) + 16:.1f}" class="candidate-label">候选低 {candidate_low:.2f}</text>',
            ]
        )
    if observation_chain:
        chain_levels = observation_chain.get("levels", [])
        for item in chain_levels[-4:]:
            level = float(item["level"])
            elements.extend(
                [
                    f'<line x1="{left_pad}" y1="{y(level):.1f}" x2="{width - right_pad}" y2="{y(level):.1f}" class="chain-line"/>',
                    f'<text x="{left_pad + 8}" y="{y(level) - 6:.1f}" class="chain-label">观察位 {level:.2f}</text>',
                ]
            )
    oi_points = []
    for idx, row in data.iterrows():
        open_ = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        volume = float(row["volume"])
        open_interest = float(row["open_interest"])
        x = x_at(idx)
        body_y = min(y(open_), y(close))
        body_h = max(2, abs(y(open_) - y(close)))
        cls = "up" if close >= open_ else "down"
        date_label = pd.to_datetime(row["datetime"]).strftime("%m-%d")
        dt = pd.to_datetime(row["datetime"]).strftime("%Y-%m-%d %H:%M")
        vol_top = volume_y(volume)
        vol_h = sub_top + sub_h - vol_top
        oi_points.append(f"{x:.1f},{oi_y(open_interest):.1f}")
        elements.extend(
            [
                f'<line x1="{x:.1f}" y1="{y(high):.1f}" x2="{x:.1f}" y2="{y(low):.1f}" class="{cls} wick"/>',
                f'<rect x="{x - candle_w / 2:.1f}" y="{body_y:.1f}" width="{candle_w:.1f}" height="{body_h:.1f}" class="{cls} body"/>',
                f'<rect x="{x - candle_w / 2:.1f}" y="{vol_top:.1f}" width="{candle_w:.1f}" height="{vol_h:.1f}" class="volume-bar"/>',
                (
                    f'<rect x="{x - step / 2:.1f}" y="{top_pad}" width="{step:.1f}" height="{sub_top + sub_h - top_pad}" '
                    f'class="candle-hit" data-chart="{escape(chart_id)}" data-time="{escape(dt)}" '
                    f'data-open="{open_:.2f}" data-high="{high:.2f}" data-low="{low:.2f}" data-close="{close:.2f}" '
                    f'data-volume="{int(volume)}" data-open-interest="{int(open_interest)}">'
                    f'<title>{escape(dt)} 开 {open_:.2f} 高 {high:.2f} 低 {low:.2f} 收 {close:.2f} 量 {int(volume)} 持仓 {int(open_interest)}</title>'
                    f'</rect>'
                ),
            ]
        )
        label_step = max(1, len(data) // 5)
        if idx % label_step == 0 or idx == len(data) - 1:
            elements.append(f'<text x="{x:.1f}" y="{height - bottom_pad + 18}" class="date-label">{date_label}</text>')

    high_x = x_at(int(data.index.get_loc(high_row.name)))
    low_x = x_at(int(data.index.get_loc(low_row.name)))
    elements.extend(
        [
            f'<circle cx="{high_x:.1f}" cy="{y(float(high_row["high"])):.1f}" r="4" class="price-marker"/>',
            f'<text x="{high_x + 8:.1f}" y="{y(float(high_row["high"])) - 8:.1f}" class="price-label">高 {float(high_row["high"]):.2f}</text>',
            f'<circle cx="{low_x:.1f}" cy="{y(float(low_row["low"])):.1f}" r="4" class="price-marker"/>',
            f'<text x="{low_x + 8:.1f}" y="{y(float(low_row["low"])) + 18:.1f}" class="price-label">低 {float(low_row["low"]):.2f}</text>',
            f'<polyline points="{" ".join(oi_points)}" fill="none" class="oi-line"/>',
            "</svg>",
        ]
    )
    return "".join(elements)


def build_frames_by_symbol(data: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    frames_by_symbol: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in SYMBOLS:
        daily = data["daily"][data["daily"]["symbol"] == symbol]
        hourly = data["hourly"][data["hourly"]["symbol"] == symbol]
        if daily.empty:
            continue
        frames_by_symbol[symbol] = build_timeframes(daily, hourly)
    return frames_by_symbol


def build_event_queue(frames_by_symbol: dict[str, dict[str, pd.DataFrame]]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for symbol, frames in frames_by_symbol.items():
        for timeframe, frame in frames.items():
            data = frame.sort_values("datetime").reset_index(drop=True)
            if data.empty:
                continue
            anchor = data.loc[data["volume"].idxmax()]
            profile = anchor_event_profile(data, anchor)
            events.append(
                {
                    "symbol": symbol,
                    "name": SYMBOLS[symbol]["name"],
                    "timeframe": timeframe,
                    "time": anchor["datetime"],
                    "volume": float(anchor["volume"]),
                    "low": float(anchor["low"]),
                    "high": float(anchor["high"]),
                    "text": profile["text"],
                }
            )
    events.sort(key=lambda item: float(item["volume"]), reverse=True)
    return events


def build_selection_module(
    priorities: dict[str, dict[str, object]],
    frames_by_symbol: dict[str, dict[str, pd.DataFrame]],
) -> str:
    if not priorities:
        return ""
    rows = sorted(priorities.values(), key=lambda item: int(item["rank"]))
    leader = rows[0]
    event_items = []
    for event in build_event_queue(frames_by_symbol)[:4]:
        event_items.append(
            f"""
            <section class="selection-card event-card">
              <b>{escape(str(event['name']))}({escape(str(event['symbol']))}) · {escape(str(event['timeframe']))}</b>
              <p>{escape(format_dt(event['time']))}，成交量 {float(event['volume']):.0f}，区间 {float(event['low']):.2f}-{float(event['high']):.2f}。</p>
            </section>
            """
        )
    cards = []
    for row in rows:
        cards.append(
            f"""
            <section class="selection-card">
              <b>{escape(str(row['priority']))} · {escape(str(row['name']))}({escape(str(row['symbol']))})</b>
              <p>{escape(str(row['summary']))}</p>
            </section>
            """
        )
    return f"""
      <section class="selection-module">
        <div class="selection-head">
          <div>
            <h2>品种选择工作流</h2>
            <p>当前最优先观察 {escape(str(leader['name']))}({escape(str(leader['symbol']))})：先选强弱，再看观察位，再等反抽/回踩，最后看风险收益比。</p>
          </div>
          <div class="workflow-steps">
            <span>1 选品</span>
            <span>2 定位</span>
            <span>3 等待</span>
            <span>4 准备</span>
          </div>
        </div>
        <div class="selection-columns">
          <div>
            <h3>强弱排序</h3>
            <div class="selection-grid">{''.join(cards)}</div>
          </div>
          <div>
            <h3>跨周期放量事件</h3>
            <div class="selection-grid event-grid">{''.join(event_items)}</div>
          </div>
        </div>
      </section>
    """


def timeframe_sections(
    frames_by_symbol: dict[str, dict[str, pd.DataFrame]],
    priorities: dict[str, dict[str, object]],
) -> str:
    sections = []
    for timeframe in TIMEFRAMES:
        cards = []
        for symbol, meta in SYMBOLS.items():
            frames = frames_by_symbol.get(symbol)
            if not frames or timeframe not in frames:
                continue
            cards.append(period_card(symbol, meta["name"], timeframe, frames[timeframe], priorities.get(symbol)))
        if cards:
            sections.append(f'<h2 class="period-group-title">{escape(timeframe)}观察</h2>')
            sections.extend(cards)
    return "\n".join(sections)


def render() -> str:
    data = load_market_data(DATA_FILES)
    frames_by_symbol = build_frames_by_symbol(data)
    priorities = build_symbol_priorities(frames_by_symbol)
    blocks = build_selection_module(priorities, frames_by_symbol) + timeframe_sections(frames_by_symbol, priorities)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>四维分析系统初版</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #090d12;
      --panel: #111821;
      --line: #2b3442;
      --text: #e6edf3;
      --muted: #9aa7b2;
      --warn: #f2c94c;
      --blue: #4cc9f0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Microsoft YaHei", "Segoe UI", system-ui, sans-serif;
      line-height: 1.55;
    }}
    header {{
      padding: 18px 28px 12px;
      border-bottom: 1px solid var(--line);
      background: #0e141c;
    }}
    h1 {{ margin: 0 0 4px; font-size: 24px; }}
    .note {{ margin: 0; color: var(--muted); font-size: 13px; }}
    main {{
      padding: 18px 28px 44px;
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 14px;
    }}
    .period-group-title {{
      grid-column: 1 / -1;
      margin: 10px 0 0;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0e141c;
      font-size: 18px;
    }}
    .selection-module {{
      grid-column: 1 / -1;
      border: 1px solid var(--line);
      background: #0e141c;
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .selection-head {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: start;
      margin-bottom: 12px;
    }}
    .selection-module h2 {{
      margin: 0 0 5px;
      font-size: 18px;
    }}
    .selection-module h3 {{
      margin: 0 0 8px;
      color: #c9d1d9;
      font-size: 13px;
    }}
    .selection-module p {{
      margin: 0;
      color: #d7dee8;
      font-size: 13px;
    }}
    .workflow-steps {{
      display: grid;
      grid-template-columns: repeat(4, max-content);
      gap: 6px;
      white-space: nowrap;
    }}
    .workflow-steps span {{
      border: 1px solid #314055;
      background: #111821;
      border-radius: 999px;
      color: #d7dee8;
      padding: 4px 9px;
      font-size: 12px;
    }}
    .selection-columns {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1.4fr);
      gap: 14px;
    }}
    .selection-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 8px;
    }}
    .event-grid {{
      grid-template-columns: minmax(0, 1fr);
    }}
    .selection-card {{
      border: 1px solid #314055;
      background: #111821;
      border-radius: 6px;
      padding: 7px 10px;
    }}
    .selection-card b {{
      color: var(--blue);
      font-size: 14px;
    }}
    .selection-card p {{
      margin: 6px 0 0;
      color: #c9d1d9;
      font-size: 12px;
    }}
    .event-card {{
      display: grid;
      grid-template-columns: 160px minmax(0, 1fr);
      gap: 10px;
      align-items: baseline;
    }}
    .event-card p {{
      margin: 0;
    }}
    .period-card {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }}
    .card-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .card-head h3 {{ margin: 0; font-size: 17px; }}
    .card-head span {{
      color: var(--warn);
      border: 1px solid #4b3f18;
      background: #17140a;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      white-space: nowrap;
    }}
    .tag-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 10px 0 12px;
    }}
    .tag {{
      border: 1px solid #314055;
      background: #121d2a;
      color: #d7dee8;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
    }}
    .period-layout {{
      display: grid;
      grid-template-columns: minmax(0, 7fr) minmax(340px, 3fr);
      gap: 16px;
      align-items: start;
    }}
    .chart-pane,
    .info-pane {{
      min-width: 0;
    }}
    .info-pane {{
      border-left: 1px solid var(--line);
      padding-left: 16px;
    }}
    .line-item {{
      border-top: 1px solid var(--line);
      padding: 9px 0 0;
      margin-top: 9px;
      display: grid;
      grid-template-columns: 86px minmax(0, 1fr);
      gap: 10px;
    }}
    .line-item b {{
      color: var(--blue);
      font-size: 12px;
      line-height: 1.35;
    }}
    .line-item b span {{
      display: inline-grid;
      place-items: center;
      width: 20px;
      height: 20px;
      margin-right: 6px;
      border-radius: 999px;
      background: #102536;
      border: 1px solid #25516d;
      color: #e6edf3;
    }}
    .line-item p {{ margin: 0; color: #d7dee8; font-size: 12px; line-height: 1.55; }}
    .kline {{
      width: 100%;
      height: 560px;
      background: #0b1118;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    .axis {{ stroke: #303846; stroke-width: 1; }}
    .axis-label, .date-label {{ fill: #9aa7b2; font-size: 12px; }}
    .date-label {{ text-anchor: middle; }}
    .up.wick {{ stroke: #ff5f56; stroke-width: 2; }}
    .down.wick {{ stroke: #2fbf71; stroke-width: 2; }}
    .up.body {{ fill: #ff5f56; }}
    .down.body {{ fill: #2fbf71; }}
    .volume-bar {{ fill: #607086; opacity: 0.78; }}
    .oi-line {{ stroke: #f2c94c; stroke-width: 2; }}
    .price-marker {{ fill: #f2c94c; stroke: #0b1118; stroke-width: 2; }}
    .price-label, .max-volume-label {{ fill: #e6edf3; font-size: 12px; font-weight: 600; }}
    .max-volume-line {{ stroke: #f2c94c; stroke-width: 1.4; stroke-dasharray: 6 4; opacity: 0.88; }}
    .candidate-line {{ stroke: #4cc9f0; stroke-width: 1.4; stroke-dasharray: 4 4; opacity: 0.92; }}
    .candidate-label {{ fill: #4cc9f0; font-size: 12px; font-weight: 700; }}
    .chain-line {{ stroke: #ff7b72; stroke-width: 1.5; stroke-dasharray: 8 4; opacity: 0.9; }}
    .chain-label {{ fill: #ffb3ad; font-size: 12px; font-weight: 700; }}
    .kline line, .kline polyline, .kline text, .kline circle, .kline rect:not(.candle-hit) {{ pointer-events: none; }}
    .candle-hit {{ fill: transparent; cursor: pointer; pointer-events: all; }}
    .candle-hit:hover {{ fill: rgba(242, 201, 76, 0.08); }}
    .kline-detail {{
      min-height: 38px;
      border: 1px solid var(--line);
      background: #0e151e;
      border-radius: 6px;
      padding: 9px 10px;
      margin-top: 8px;
      color: #c9d1d9;
      font-size: 13px;
    }}
    .kline-detail b {{ color: #e6edf3; margin-right: 10px; }}
    .kline-detail span {{ display: inline-block; margin-right: 12px; }}
    .level-0 {{ border-left: 4px solid #7b8491; }}
    .level-1 {{ border-left: 4px solid #4cc9f0; }}
    .level-2 {{ border-left: 4px solid #f2c94c; }}
    .level-3 {{ border-left: 4px solid #ff7b72; }}
    @media (max-width: 900px) {{
      main {{
        padding: 14px;
      }}
      .selection-head,
      .selection-columns {{
        grid-template-columns: 1fr;
        display: grid;
      }}
      .workflow-steps {{
        grid-template-columns: repeat(2, max-content);
      }}
      .period-layout {{
        grid-template-columns: 1fr;
      }}
      .selection-grid {{
        grid-template-columns: 1fr;
      }}
      .info-pane {{
        border-left: 0;
        border-top: 1px solid var(--line);
        padding-left: 0;
        padding-top: 12px;
      }}
      .line-item {{
        grid-template-columns: 1fr;
        gap: 5px;
      }}
      .kline {{
        height: 440px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>四维分析系统初版</h1>
    <p class="note">减法版：每个品种、每个周期只输出当前状态、当前观察位、等待原因、准备等级、失效条件。</p>
  </header>
  <main>
    {blocks}
  </main>
  <script>
    document.addEventListener("click", function (event) {{
      const target = event.target.closest(".candle-hit");
      if (!target) return;
      const detail = document.getElementById("detail-" + target.dataset.chart);
      if (!detail) return;
      detail.innerHTML = [
        "<b>" + target.dataset.time + "</b>",
        "<span>开盘 " + target.dataset.open + "</span>",
        "<span>高点 " + target.dataset.high + "</span>",
        "<span>低点 " + target.dataset.low + "</span>",
        "<span>收盘 " + target.dataset.close + "</span>",
        "<span>成交量 " + Number(target.dataset.volume).toLocaleString("zh-CN") + "</span>",
        "<span>持仓量 " + Number(target.dataset.openInterest).toLocaleString("zh-CN") + "</span>"
      ].join("");
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    html = render()
    ROOT.mkdir(parents=True, exist_ok=True)
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "index.html").write_text(html, encoding="utf-8")
    (DOCS_ROOT / "index.html").write_text(html, encoding="utf-8")
    print(f"生成完成: {ROOT / 'index.html'}")
    print(f"发布同步: {DOCS_ROOT / 'index.html'}")


if __name__ == "__main__":
    main()
