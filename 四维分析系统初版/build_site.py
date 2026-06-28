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


def period_card(symbol: str, name: str, timeframe: str, frame: pd.DataFrame) -> str:
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

    if latest_close < anchor_low:
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
    tags.append(anchor_state["state"])
    readiness_level, readiness_text = readiness_from_tags(tags, pressure_distance, support_distance)
    wait_reason = waiting_reason_from_state(tags, pressure_distance, support_distance, readiness_level)
    wait_reason = combine_wait_reason(wait_reason, anchor_state)
    status = build_status(tags, close_change, oi_change, below_ratio, above_ratio, inside_ratio)
    observe = (
        f"压力 {pressure:.2f}，距最新收盘 {pressure_distance:.2f}%；"
        f"支撑 {support:.2f}，距最新收盘 {support_distance:.2f}%。"
        f"最大量K线 {format_dt(anchor['datetime'])}，区间 {anchor_low:.2f}-{anchor_high:.2f}。"
        f"{anchor_state['observe']}"
    )
    invalidation = build_invalidation(tags, pressure, support, anchor_high, anchor_low, anchor_state)
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
            {kline_svg(data, chart_id, anchor_state)}
            <div class="kline-detail" id="detail-{escape(chart_id)}">点击K线查看该根K线的时间、开高低收、成交量、持仓量。</div>
          </div>
          <aside class="info-pane">
            <div class="tag-row">{tag_html}</div>
            {line_item("1. 当前状态", status)}
            {line_item("2. 当前观察位", observe)}
            {line_item("3. 等待原因", wait_reason)}
            {line_item("4. 准备等级", readiness_text)}
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
) -> str:
    if "支撑转压力" in tags:
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
        f"收在区间内占比 {inside_ratio * 100:.1f}%。{attitude}"
    )


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


def build_invalidation(
    tags: list[str],
    pressure: float,
    support: float,
    anchor_high: float,
    anchor_low: float,
    anchor_state: dict[str, object],
) -> str:
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
    return f"""
      <section class="line-item">
        <b>{escape(title)}</b>
        <p>{escape(body)}</p>
      </section>
    """


def kline_svg(
    frame: pd.DataFrame,
    chart_id: str,
    anchor_state: dict[str, object] | None = None,
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


def timeframe_sections(frames_by_symbol: dict[str, dict[str, pd.DataFrame]]) -> str:
    sections = []
    for timeframe in TIMEFRAMES:
        cards = []
        for symbol, meta in SYMBOLS.items():
            frames = frames_by_symbol.get(symbol)
            if not frames or timeframe not in frames:
                continue
            cards.append(period_card(symbol, meta["name"], timeframe, frames[timeframe]))
        if cards:
            sections.append(f'<h2 class="period-group-title">{escape(timeframe)}对比</h2>')
            sections.extend(cards)
    return "\n".join(sections)


def render() -> str:
    data = load_market_data(DATA_FILES)
    blocks = timeframe_sections(build_frames_by_symbol(data))
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
      padding: 24px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: #0e141c;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .note {{ margin: 0; color: var(--muted); }}
    main {{
      padding: 18px 28px 44px;
      display: grid;
      grid-template-columns: repeat(2, minmax(680px, 1fr));
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
      grid-template-columns: minmax(0, 3fr) minmax(230px, 1fr);
      gap: 14px;
      align-items: start;
    }}
    .chart-pane,
    .info-pane {{
      min-width: 0;
    }}
    .info-pane {{
      border-left: 1px solid var(--line);
      padding-left: 14px;
    }}
    .line-item {{
      border-top: 1px solid var(--line);
      padding: 8px 0 0;
      margin-top: 8px;
    }}
    .line-item b {{ color: var(--blue); }}
    .line-item p {{ margin: 5px 0 0; color: #d7dee8; font-size: 13px; }}
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
    @media (max-width: 1480px) {{
      main {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 900px) {{
      main {{
        padding: 14px;
      }}
      .period-layout {{
        grid-template-columns: 1fr;
      }}
      .info-pane {{
        border-left: 0;
        border-top: 1px solid var(--line);
        padding-left: 0;
        padding-top: 12px;
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
