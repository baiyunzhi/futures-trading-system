from __future__ import annotations

from pathlib import Path
from html import escape

import pandas as pd

from market_system import (
    SYMBOLS,
    TIMEFRAMES,
    analyze_market,
    build_timeframes,
    current_pressure_level,
    format_dt,
    load_market_data,
)


ROOT = Path(__file__).resolve().parent
DATA_FILES = {
    "RB0": {
        "daily": ROOT / "rb_recent_daily.csv",
        "hourly": ROOT / "rb_recent_hourly.csv",
    },
    "V0": {
        "daily": ROOT / "v_recent_daily.csv",
        "hourly": ROOT / "v_recent_hourly.csv",
    },
}
WEB_PATH = ROOT / "web" / "index.html"


def build_comparison_module(reports, data: dict[str, pd.DataFrame]) -> str:
    cards = []
    snapshots = []
    reports_by_symbol = {report.symbol: report for report in reports}
    for symbol, meta in SYMBOLS.items():
        report = reports_by_symbol.get(symbol)
        if report is None:
            continue
        daily = data["daily"]
        hourly = data["hourly"]
        frames = build_timeframes(daily[daily["symbol"] == symbol], hourly[hourly["symbol"] == symbol])
        snapshot = build_symbol_snapshot(symbol, meta["name"], frames)
        snapshots.append(snapshot)
        cards.append(comparison_card(snapshot))
    if not cards:
        return ""
    conclusion = comparison_conclusion(snapshots)
    return f"""
    <section class="comparison">
      <div class="comparison-head">
        <h2>品种对比</h2>
        <p>{escape(conclusion)}</p>
      </div>
      <div class="comparison-grid">
        {"".join(cards)}
      </div>
    </section>
    """


def build_symbol_snapshot(symbol: str, name: str, frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    timeframe_rows = [timeframe_score(timeframe, frames[timeframe]) for timeframe in TIMEFRAMES if timeframe in frames]
    short_score = sum(float(row["short_score"]) for row in timeframe_rows)
    long_score = sum(float(row["long_score"]) for row in timeframe_rows)
    daily = next(row for row in timeframe_rows if row["timeframe"] == "日线")
    hourly = next(row for row in timeframe_rows if row["timeframe"] == "小时线")
    if short_score > long_score + 3:
        control = "空方控制"
    elif long_score > short_score + 3:
        control = "多方控制"
    else:
        control = "多空拉锯"
    distance = min(float(daily["pressure_distance_pct"]), float(hourly["pressure_distance_pct"]))
    if control == "空方控制" and distance <= 2:
        action = "优先观察"
        reason = "价格接近支撑转压力位，适合等待反抽失败确认。"
    elif control == "空方控制":
        action = "等待反抽"
        reason = "空方控制明确，但当前价格离压力位较远，追空位置不好。"
    elif control == "多方控制":
        action = "等待回踩"
        reason = "多方控制占优，等待回踩支撑后再观察是否守住。"
    else:
        action = "等待方向"
        reason = "多空还在区间内拉锯，支撑压力没有给出明确转换。"
    return {
        "symbol": symbol,
        "name": name,
        "control": control,
        "action": action,
        "reason": reason,
        "short_score": short_score,
        "long_score": long_score,
        "timeframes": timeframe_rows,
        "distance": distance,
    }


def timeframe_score(timeframe: str, frame: pd.DataFrame) -> dict[str, object]:
    data = frame.sort_values("datetime").reset_index(drop=True)
    latest = data.iloc[-1]
    first = data.iloc[0]
    anchor = data.loc[data["volume"].idxmax()]
    after = data[data["datetime"] > anchor["datetime"]]
    high = float(anchor["high"])
    low = float(anchor["low"])
    close = float(latest["close"])
    close_change = close - float(first["close"])
    oi_change = float(latest["open_interest"] - first["open_interest"])
    pressure = high
    support = low
    if not after.empty and close < low:
        pressure = current_pressure_level(after, low)
    elif not after.empty and close > high:
        support = high
    pressure_distance_pct = abs(pressure - close) / close * 100 if close else 999.0
    support_distance_pct = abs(close - support) / close * 100 if close else 999.0
    below_ratio = float((after["close"] < low).sum() / len(after)) if len(after) else 0.0
    above_ratio = float((after["close"] > high).sum() / len(after)) if len(after) else 0.0
    inside_ratio = float(((after["close"] >= low) & (after["close"] <= high)).sum() / len(after)) if len(after) else 0.0
    short_score = 0.0
    long_score = 0.0
    if close < low:
        short_score += 3
    elif close > high:
        long_score += 3
    else:
        short_score += 0.5
        long_score += 0.5
    short_score += below_ratio * 3
    long_score += above_ratio * 3
    if close_change < 0 and oi_change > 0:
        short_score += 3
    elif close_change > 0 and oi_change > 0:
        long_score += 3
    elif close_change < 0 and oi_change < 0:
        short_score += 1
    elif close_change > 0 and oi_change < 0:
        long_score += 1
    if inside_ratio >= 0.45:
        short_score -= 0.5
        long_score -= 0.5
    return {
        "timeframe": timeframe,
        "anchor_time": format_dt(anchor["datetime"]),
        "anchor_low": low,
        "anchor_high": high,
        "latest_close": close,
        "pressure": pressure,
        "support": support,
        "pressure_distance_pct": pressure_distance_pct,
        "support_distance_pct": support_distance_pct,
        "below_ratio": below_ratio,
        "above_ratio": above_ratio,
        "inside_ratio": inside_ratio,
        "close_change": close_change,
        "oi_change": oi_change,
        "short_score": short_score,
        "long_score": long_score,
    }


def comparison_card(snapshot: dict[str, object]) -> str:
    rows = []
    for row in snapshot["timeframes"]:
        rows.append(
            "<li>"
            f"{escape(str(row['timeframe']))}：最大量K线 {escape(str(row['anchor_time']))}，"
            f"区间 {float(row['anchor_low']):.2f}-{float(row['anchor_high']):.2f}，"
            f"最新收盘 {float(row['latest_close']):.2f}，"
            f"收在锚点低点下方占比 {float(row['below_ratio']) * 100:.1f}%，"
            f"持仓变化 {float(row['oi_change']):+.0f}，"
            f"距离当前压力位 {float(row['pressure_distance_pct']):.2f}%。"
            "</li>"
        )
    return f"""
      <article class="comparison-card">
        <div class="comparison-title">
          <h3>{escape(str(snapshot["name"]))}({escape(str(snapshot["symbol"]))})</h3>
          <span>{escape(str(snapshot["control"]))}</span>
        </div>
        <p class="comparison-action">{escape(str(snapshot["action"]))}</p>
        <p>{escape(str(snapshot["reason"]))}</p>
        <ul>{''.join(rows)}</ul>
      </article>
    """


def comparison_conclusion(snapshots: list[dict[str, object]]) -> str:
    if len(snapshots) < 2:
        return "当前只有一个品种，系统先给出单品种观察状态。"
    strongest = max(snapshots, key=lambda item: float(item["short_score"]))
    nearest = min(snapshots, key=lambda item: float(item["distance"]))
    return (
        f"当前对比结果：{strongest['name']}({strongest['symbol']}) 空方控制更强；"
        f"{nearest['name']}({nearest['symbol']}) 距离压力观察位更近，更适合优先盯反抽确认。"
        "若方向明确但价格远离压力位，系统归为等待，不追。"
    )


def build_timeframe_sections(reports, data: dict[str, pd.DataFrame]) -> str:
    blocks = []
    reports_by_symbol = {report.symbol: report for report in reports}
    for symbol, meta in SYMBOLS.items():
        report = reports_by_symbol.get(symbol)
        if report is None:
            continue
        daily = data["daily"]
        hourly = data["hourly"]
        frames = build_timeframes(daily[daily["symbol"] == symbol], hourly[hourly["symbol"] == symbol])
        for timeframe in TIMEFRAMES:
            frame = frames[timeframe].copy()
            view = report.views[timeframe]
            chart_id = f"{symbol}-{timeframe}"
            blocks.append(
                f"""
                <section class="period">
                  <div class="period-head">
                    <h2>{meta["name"]}({symbol}) · {timeframe}</h2>
                    <p>{view.summary}</p>
                    {observation_list(view.observations)}
                    {workflow_alerts(view.workflow_alerts)}
                  </div>
                  {kline_svg(frame, chart_id)}
                  <div class="kline-detail" id="detail-{chart_id}">点击K线查看该根K线的时间、开高低收、成交量、持仓量。</div>
                </section>
                """
            )
    return "\n".join(blocks)


def observation_list(items: list[str]) -> str:
    rows = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f'<ul class="observations">{rows}</ul>'


def workflow_alerts(items: list[str]) -> str:
    if not items:
        return ""
    rows = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f'<div class="workflow"><h3>系统自动提示</h3><ul>{rows}</ul></div>'


def kline_svg(frame: pd.DataFrame, chart_id: str, width: int = 1100, height: int = 520) -> str:
    data = frame.copy()
    if data.empty:
        return ""
    top_pad = 24
    price_h = 260
    sub_top = 330
    sub_h = 120
    bottom_pad = 32
    left_pad = 58
    right_pad = 18
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
    candle_w = max(10, step * 0.48)

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
        f'<text x="{width - right_pad - 220}" y="{y(max_volume_high) - 6:.1f}" class="max-volume-label">最大成交量K线高点 {max_volume_high:.2f}</text>',
        f'<text x="{width - right_pad - 220}" y="{y(max_volume_low) + 16:.1f}" class="max-volume-label">最大成交量K线低点 {max_volume_low:.2f}</text>',
        f'<text x="8" y="{sub_top + 12}" class="axis-label">成交量/持仓量</text>',
        f'<line x1="{left_pad}" y1="{sub_top + sub_h}" x2="{width - right_pad}" y2="{sub_top + sub_h}" class="axis"/>',
    ]
    oi_points = []
    for idx, (_, row) in enumerate(data.iterrows()):
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
        vol_top = volume_y(volume)
        vol_h = sub_top + sub_h - vol_top
        oi_points.append(f"{x:.1f},{oi_y(open_interest):.1f}")
        dt = pd.to_datetime(row["datetime"]).strftime("%Y-%m-%d %H:%M")
        elements.extend([
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
        ])
        label_step = max(1, len(data) // 8)
        if idx % label_step == 0 or idx == len(data) - 1:
            elements.append(f'<text x="{x:.1f}" y="{height - bottom_pad + 18}" class="date-label">{date_label}</text>')
    high_pos = data.index.get_loc(high_row.name)
    low_pos = data.index.get_loc(low_row.name)
    high_x = x_at(high_pos)
    low_x = x_at(low_pos)
    elements.extend([
        f'<circle cx="{high_x:.1f}" cy="{y(float(high_row["high"])):.1f}" r="4" class="price-marker"/>',
        f'<text x="{high_x + 8:.1f}" y="{y(float(high_row["high"])) - 8:.1f}" class="price-label">高点 {float(high_row["high"]):.2f}</text>',
        f'<circle cx="{low_x:.1f}" cy="{y(float(low_row["low"])):.1f}" r="4" class="price-marker"/>',
        f'<text x="{low_x + 8:.1f}" y="{y(float(low_row["low"])) + 18:.1f}" class="price-label">低点 {float(low_row["low"]):.2f}</text>',
    ])
    elements.append(f'<polyline points="{" ".join(oi_points)}" fill="none" class="oi-line"/>')
    elements.append("</svg>")
    return "".join(elements)


def render_html(reports, data: dict[str, pd.DataFrame]) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>螺纹钢与PVC最近两个月四维行情描述</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0d10;
      --panel: #151922;
      --panel2: #10141b;
      --line: #2b3240;
      --text: #e6edf3;
      --muted: #8b949e;
      --good: #2fbf71;
      --bad: #ff5f56;
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
      padding: 24px 28px 10px;
      border-bottom: 1px solid var(--line);
      background: #0f131a;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 0 0 8px; font-size: 18px; }}
    main {{ padding: 18px 28px 40px; }}
    .note {{ color: var(--muted); margin: 0; }}
    .card, .period {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .period-head p {{ margin: 0 0 14px; color: #c9d1d9; }}
    .observations {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 8px;
      padding: 0;
      margin: 14px 0 0;
      list-style: none;
    }}
    .observations li {{
      border: 1px solid var(--line);
      background: #10141b;
      border-radius: 6px;
      padding: 10px 12px;
      color: #d7dee8;
      font-size: 13px;
      line-height: 1.65;
    }}
    .workflow {{
      border: 1px solid #4b3f18;
      background: #17140a;
      border-radius: 6px;
      padding: 12px 14px;
      margin: 14px 0 0;
    }}
    .workflow h3 {{
      margin: 0 0 8px;
      color: var(--warn);
      font-size: 15px;
    }}
    .workflow ul {{
      padding-left: 18px;
      margin: 0;
    }}
    .workflow li {{
      margin: 6px 0;
      color: #eadfbd;
      font-size: 13px;
      line-height: 1.7;
    }}
    .comparison {{
      background: #121821;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .comparison-head h2 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    .comparison-head p {{
      margin: 0 0 14px;
      color: #d7dee8;
    }}
    .comparison-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 12px;
    }}
    .comparison-card {{
      border: 1px solid var(--line);
      background: #0f141b;
      border-radius: 6px;
      padding: 14px;
    }}
    .comparison-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .comparison-title h3 {{
      margin: 0;
      font-size: 16px;
    }}
    .comparison-title span {{
      border: 1px solid #4b3f18;
      background: #17140a;
      color: var(--warn);
      border-radius: 999px;
      padding: 3px 10px;
      font-size: 12px;
      white-space: nowrap;
    }}
    .comparison-action {{
      color: var(--blue);
      font-weight: 700;
      margin: 0 0 6px;
    }}
    .comparison-card p {{
      color: #c9d1d9;
      margin: 0 0 10px;
    }}
    .comparison-card ul {{
      padding-left: 18px;
      margin: 0;
    }}
    .comparison-card li {{
      margin: 6px 0;
      color: #d7dee8;
      font-size: 13px;
      line-height: 1.65;
    }}
    .kline {{
      width: 100%;
      height: 520px;
      background: #0d1117;
      border: 1px solid var(--line);
      border-radius: 6px;
      margin-bottom: 12px;
    }}
    .axis {{ stroke: #303846; stroke-width: 1; }}
    .axis-label, .date-label {{ fill: #8b949e; font-size: 12px; }}
    .date-label {{ text-anchor: middle; }}
    .up.wick {{ stroke: #ff5f56; stroke-width: 2; }}
    .down.wick {{ stroke: #2fbf71; stroke-width: 2; }}
    .up.body {{ fill: #ff5f56; }}
    .down.body {{ fill: #2fbf71; }}
    .volume-bar {{ fill: #607086; opacity: 0.78; }}
    .oi-line {{ stroke: #f2c94c; stroke-width: 2; }}
    .price-marker {{ fill: #f2c94c; stroke: #0d1117; stroke-width: 2; }}
    .price-label, .max-volume-label {{ fill: #e6edf3; font-size: 12px; font-weight: 600; }}
    .max-volume-line {{ stroke: #f2c94c; stroke-width: 1.4; stroke-dasharray: 6 4; opacity: 0.88; }}
    .kline line, .kline polyline, .kline text, .kline circle, .kline rect:not(.candle-hit) {{ pointer-events: none; }}
    .candle-hit {{ fill: transparent; cursor: pointer; pointer-events: all; }}
    .candle-hit:hover {{ fill: rgba(242, 201, 76, 0.08); }}
    .kline-detail {{
      min-height: 42px;
      border: 1px solid var(--line);
      background: #10141b;
      border-radius: 6px;
      padding: 10px 12px;
      color: #c9d1d9;
      font-size: 13px;
    }}
    .kline-detail b {{ color: #e6edf3; margin-right: 10px; }}
    .kline-detail span {{ display: inline-block; margin-right: 14px; }}
    .steps li {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <header>
    <h1>螺纹钢与PVC最近两个月四维行情描述</h1>
    <p class="note">螺纹钢和PVC最近两个月真实数据。日线、周线、小时线分别独立描述，只落地K线、成交量、持仓量、时间周期四个维度。</p>
  </header>
  <main>
    <section class="card">
      <h2>一步一步的流程</h2>
      <ol class="steps">
        <li>数据窗口：螺纹钢和PVC最近两个月真实日线和小时线数据。</li>
        <li>日线、周线、小时线分别独立描述，周线由日线聚合。</li>
        <li>价格用K线高点、低点、收盘位置描述波动规律。</li>
        <li>成交量柱和持仓量线放在同一个区域显示。</li>
        <li>K线图标出两个月高低点价格，并标出最大成交量K线的高低点。</li>
        <li>点击任意K线显示该根K线的时间、开高低收、成交量、持仓量。</li>
        <li>K线颜色：红涨绿跌。</li>
        <li>每个品种、每个周期独立执行同一套支撑压力演变工作流：最大成交量K线作为观察锚点，后续突破、跌破、假突破、回抽失败、支撑转压力或压力转支撑都自动写入行情表述。</li>
      </ol>
    </section>

    {build_comparison_module(reports, data)}

    {build_timeframe_sections(reports, data)}
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
    data = load_market_data(DATA_FILES)
    reports = analyze_market(data)
    WEB_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEB_PATH.write_text(render_html(reports, data), encoding="utf-8")
    print(f"生成完成: {WEB_PATH}")


if __name__ == "__main__":
    main()
