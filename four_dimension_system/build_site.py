from __future__ import annotations

from pathlib import Path
from html import escape

import pandas as pd

from market_system import SYMBOLS, TIMEFRAMES, analyze_market, build_timeframes, load_market_data


ROOT = Path(__file__).resolve().parent
DAILY_DATA_PATH = ROOT / "rb_recent_daily.csv"
HOURLY_DATA_PATH = ROOT / "rb_recent_hourly.csv"
WEB_PATH = ROOT / "web" / "index.html"


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
  <title>螺纹钢最近两个月四维行情描述</title>
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
    <h1>螺纹钢最近两个月四维行情描述</h1>
    <p class="note">螺纹钢最近两个月真实数据。日线、周线、小时线分别独立描述，只落地K线、成交量、持仓量、时间周期四个维度。</p>
  </header>
  <main>
    <section class="card">
      <h2>一步一步的流程</h2>
      <ol class="steps">
        <li>数据窗口：螺纹钢最近两个月真实日线和小时线数据。</li>
        <li>日线、周线、小时线分别独立描述，周线由日线聚合。</li>
        <li>价格用K线高点、低点、收盘位置描述波动规律。</li>
        <li>成交量柱和持仓量线放在同一个区域显示。</li>
        <li>K线图标出两个月高低点价格，并标出最大成交量K线的高低点。</li>
        <li>点击任意K线显示该根K线的时间、开高低收、成交量、持仓量。</li>
        <li>K线颜色：红涨绿跌。</li>
      </ol>
    </section>

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
    data = load_market_data(DAILY_DATA_PATH, HOURLY_DATA_PATH)
    reports = analyze_market(data)
    WEB_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEB_PATH.write_text(render_html(reports, data), encoding="utf-8")
    print(f"生成完成: {WEB_PATH}")


if __name__ == "__main__":
    main()
