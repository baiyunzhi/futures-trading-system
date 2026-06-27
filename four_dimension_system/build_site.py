from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_system import SYMBOLS, TIMEFRAMES, analyze_market, build_timeframes, load_or_create_data


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "sample_hourly_data.csv"
WEB_PATH = ROOT / "web" / "index.html"


def td(value: object, cls: str = "") -> str:
    klass = f' class="{cls}"' if cls else ""
    return f"<td{klass}>{value}</td>"


def build_timeframe_sections(reports, data: pd.DataFrame) -> str:
    blocks = []
    reports_by_symbol = {report.symbol: report for report in reports}
    for symbol, meta in SYMBOLS.items():
        report = reports_by_symbol.get(symbol)
        if report is None:
            continue
        frames = build_timeframes(data[data["symbol"] == symbol])
        for timeframe in TIMEFRAMES:
            frame = frames[timeframe].tail(12).copy()
            view = report.views[timeframe]
            blocks.append(
                f"""
                <section class="period">
                  <div class="period-head">
                    <h2>{meta["name"]}({symbol}) · {timeframe}</h2>
                    <p>{view.summary}</p>
                  </div>
                  {kline_svg(frame)}
                  {build_kline_table(frame)}
                </section>
                """
            )
    return "\n".join(blocks)


def build_kline_table(frame: pd.DataFrame) -> str:
    rows = []
    for _, row in frame.iterrows():
        dt = pd.to_datetime(row["datetime"]).strftime("%Y-%m-%d %H:%M")
        rows.append(
            "<tr>"
            + td(dt)
            + td(f"{float(row['open']):.2f}")
            + td(f"{float(row['high']):.2f}")
            + td(f"{float(row['low']):.2f}")
            + td(f"{float(row['close']):.2f}")
            + td(int(row["volume"]))
            + td(int(row["open_interest"]))
            + "</tr>"
        )
    return table(["时间", "开盘", "高点", "低点", "收盘", "成交量", "持仓量"], rows)


def kline_svg(frame: pd.DataFrame, width: int = 980, height: int = 360) -> str:
    data = frame.tail(12).copy()
    if data.empty:
        return ""
    top_pad = 24
    bottom_pad = 48
    left_pad = 58
    right_pad = 18
    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad
    price_high = float(data["high"].max())
    price_low = float(data["low"].min())
    price_span = price_high - price_low if price_high > price_low else 1.0
    step = plot_w / max(1, len(data))
    candle_w = max(10, step * 0.48)

    def y(price: float) -> float:
        return top_pad + (price_high - price) / price_span * plot_h

    elements = [
        f'<svg viewBox="0 0 {width} {height}" class="kline" role="img">',
        f'<line x1="{left_pad}" y1="{top_pad}" x2="{left_pad}" y2="{top_pad + plot_h}" class="axis"/>',
        f'<line x1="{left_pad}" y1="{top_pad + plot_h}" x2="{width - right_pad}" y2="{top_pad + plot_h}" class="axis"/>',
        f'<text x="8" y="{top_pad + 4}" class="axis-label">{price_high:.2f}</text>',
        f'<text x="8" y="{top_pad + plot_h}" class="axis-label">{price_low:.2f}</text>',
    ]
    for idx, (_, row) in enumerate(data.iterrows()):
        open_ = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        x = left_pad + idx * step + step / 2
        body_y = min(y(open_), y(close))
        body_h = max(2, abs(y(open_) - y(close)))
        cls = "up" if close >= open_ else "down"
        date_label = pd.to_datetime(row["datetime"]).strftime("%m-%d")
        elements.extend([
            f'<line x1="{x:.1f}" y1="{y(high):.1f}" x2="{x:.1f}" y2="{y(low):.1f}" class="{cls} wick"/>',
            f'<rect x="{x - candle_w / 2:.1f}" y="{body_y:.1f}" width="{candle_w:.1f}" height="{body_h:.1f}" class="{cls} body"/>',
        ])
        if idx % 2 == 0 or len(data) <= 8:
            elements.append(f'<text x="{x:.1f}" y="{height - 20}" class="date-label">{date_label}</text>')
    elements.append("</svg>")
    return "".join(elements)


def table(headers: list[str], rows: list[str]) -> str:
    head = "".join(f"<th>{header}</th>" for header in headers)
    body = "".join(rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_html(reports, data: pd.DataFrame) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>螺纹钢三周期客观行情描述</title>
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
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: center;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{ background: var(--panel2); color: #c9d1d9; }}
    small {{ color: var(--muted); }}
    .kline {{
      width: 100%;
      height: 360px;
      background: #0d1117;
      border: 1px solid var(--line);
      border-radius: 6px;
      margin-bottom: 12px;
    }}
    .axis {{ stroke: #303846; stroke-width: 1; }}
    .axis-label, .date-label {{ fill: #8b949e; font-size: 12px; }}
    .date-label {{ text-anchor: middle; }}
    .up.wick {{ stroke: #2fbf71; stroke-width: 2; }}
    .down.wick {{ stroke: #ff5f56; stroke-width: 2; }}
    .up.body {{ fill: #2fbf71; }}
    .down.body {{ fill: #ff5f56; }}
    .steps li {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <header>
    <h1>螺纹钢三周期客观行情描述</h1>
    <p class="note">周线、日线、小时线分别独立观察，不互相作为判断依据。只描述价格高低点、成交量、持仓量和时间节奏。</p>
  </header>
  <main>
    <section class="card">
      <h2>一步一步的流程</h2>
      <ol class="steps">
        <li>周线只描述周线自身的波动，不影响日线和小时线。</li>
        <li>日线只描述日线自身的波动，不把周线当作判断依据。</li>
        <li>小时线只描述小时线自身的波动。</li>
        <li>价格用K线高点、低点、收盘位置描述波动规律。</li>
        <li>成交量和持仓量用来描述多空双方态度和波动持续性。</li>
      </ol>
    </section>

    {build_timeframe_sections(reports, data)}
  </main>
</body>
</html>
"""


def main() -> None:
    data = load_or_create_data(DATA_PATH)
    reports = analyze_market(data)
    WEB_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEB_PATH.write_text(render_html(reports, data), encoding="utf-8")
    print(f"生成完成: {WEB_PATH}")


if __name__ == "__main__":
    main()
