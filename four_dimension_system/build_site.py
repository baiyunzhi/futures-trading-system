from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_system import DIMENSIONS, SYMBOLS, TIMEFRAMES, analyze_market, build_timeframes, load_or_create_data, sparkline_svg


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "sample_hourly_data.csv"
WEB_PATH = ROOT / "web" / "index.html"


def td(value: object, cls: str = "") -> str:
    klass = f' class="{cls}"' if cls else ""
    return f"<td{klass}>{value}</td>"


def badge(text: str) -> str:
    cls = {
        "BET": "good",
        "PLAN": "warn",
        "WAIT": "muted",
        "多": "good",
        "空": "bad",
        "观望": "muted",
        "A": "good",
        "B": "warn",
        "观察": "muted",
    }.get(str(text), "")
    return f'<span class="badge {cls}">{text}</span>'


def build_plan_table(reports) -> str:
    headers = ["品种", "动作", "方向", "等级", "触发条件", "入场", "止损", "1R", "2R", "风险", "手数", "失效", "原因"]
    rows = []
    for report in reports:
        plan = report.bet_plan
        rows.append(
            "<tr>"
            + td(f"{report.name}({report.symbol})")
            + td(badge(plan.action))
            + td(badge(plan.direction))
            + td(badge(plan.grade))
            + td(plan.trigger, "left")
            + td(plan.entry)
            + td(plan.stop)
            + td(plan.target_1r)
            + td(plan.target_2r)
            + td(f"{plan.risk_pct:.1%}")
            + td(plan.lots)
            + td(plan.invalidation, "left")
            + td(plan.reason, "left")
            + "</tr>"
        )
    return table(headers, rows)


def build_matrix_table(reports) -> str:
    headers = ["品种", "周期", *DIMENSIONS, "周期结论"]
    rows = []
    for report in reports:
        for timeframe in TIMEFRAMES:
            view = report.views[timeframe]
            rows.append(
                "<tr>"
                + td(f"{report.name}({report.symbol})")
                + td(timeframe)
                + dim_cell(view.price)
                + dim_cell(view.volume)
                + dim_cell(view.open_interest)
                + dim_cell(view.time)
                + td(view.summary, "left")
                + "</tr>"
            )
    return table(headers, rows)


def dim_cell(view) -> str:
    cls = {"bull": "good", "bear": "bad", "neutral": "muted"}.get(view.bias, "muted")
    return td(f'<b class="{cls}">{view.state}</b><br><small>{view.evidence}</small>', "left")


def build_story_cards(reports) -> str:
    cards = []
    for report in reports:
        cards.append(
            f"""
            <section class="card story">
              <h3>{report.name}({report.symbol})</h3>
              <p>{report.story}</p>
              <p><b>下注判断：</b>{report.bet_plan.reason}</p>
            </section>
            """
        )
    return "\n".join(cards)


def build_chart_section(data: pd.DataFrame) -> str:
    blocks = []
    for symbol, meta in SYMBOLS.items():
        symbol_df = data[data["symbol"] == symbol]
        frames = build_timeframes(symbol_df)
        blocks.append(
            f"""
            <section class="card chart-card">
              <h3>{meta["name"]}({symbol}) 小时收盘路径</h3>
              {sparkline_svg(frames["小时线"])}
            </section>
            """
        )
    return "\n".join(blocks)


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
  <title>四维三周期客观行情系统</title>
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
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; }}
    main {{ padding: 18px 28px 40px; }}
    .note {{ color: var(--muted); margin: 0; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }}
    .story p {{ margin: 8px 0 0; color: #c9d1d9; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 980px; }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: center;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{ background: var(--panel2); color: #c9d1d9; }}
    td.left {{ text-align: left; min-width: 180px; }}
    small {{ color: var(--muted); }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .warn {{ color: var(--warn); }}
    .muted {{ color: var(--muted); }}
    .badge {{
      display: inline-block;
      min-width: 44px;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #11161e;
    }}
    .spark {{
      width: 100%;
      height: 120px;
      background: #0d1117;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    .steps li {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <header>
    <h1>四维三周期客观行情系统</h1>
    <p class="note">只使用价格、成交量、持仓量、时间。三个周期：周线、日线、小时线。不加入其他指标。</p>
  </header>
  <main>
    <section class="card">
      <h2>一步一步的流程</h2>
      <ol class="steps">
        <li>先看周线：只判断大环境，不下注。</li>
        <li>再看日线：确定当下行情是突破、区间上沿、区间下沿，还是中部震荡。</li>
        <li>最后看小时线：只有小时线触发，才允许执行下注。</li>
        <li>下注前必须确认四个维度：价格方向、成交量主动性、持仓量持续性、时间节奏。</li>
        <li>先确定失效价，再计算手数；没有触发就只做计划，不提前预测。</li>
      </ol>
    </section>

    <section class="card">
      <h2>下注计划</h2>
      {build_plan_table(reports)}
    </section>

    <section class="card">
      <h2>三周期 × 四维度行情矩阵</h2>
      {build_matrix_table(reports)}
    </section>

    <section class="grid">
      {build_story_cards(reports)}
    </section>

    <section class="grid">
      {build_chart_section(data)}
    </section>
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
