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
    readiness_level, readiness_text = readiness_from_tags(tags, pressure_distance, support_distance)
    wait_reason = waiting_reason_from_state(tags, pressure_distance, support_distance, readiness_level)
    status = build_status(tags, close_change, oi_change, below_ratio, above_ratio, inside_ratio)
    observe = (
        f"压力 {pressure:.2f}，距最新收盘 {pressure_distance:.2f}%；"
        f"支撑 {support:.2f}，距最新收盘 {support_distance:.2f}%。"
        f"最大量K线 {format_dt(anchor['datetime'])}，区间 {anchor_low:.2f}-{anchor_high:.2f}。"
    )
    invalidation = build_invalidation(tags, pressure, support, anchor_high, anchor_low)
    tag_html = "".join(f'<span class="tag">{escape(tag)}</span>' for tag in tags)
    return f"""
      <article class="period-card level-{readiness_level}">
        <div class="card-head">
          <h3>{escape(name)}({escape(symbol)}) · {escape(timeframe)}</h3>
          <span>{escape(readiness_text)}</span>
        </div>
        <div class="tag-row">{tag_html}</div>
        {line_item("1. 当前状态", status)}
        {line_item("2. 当前观察位", observe)}
        {line_item("3. 等待原因", wait_reason)}
        {line_item("4. 准备等级", readiness_text)}
        {line_item("5. 失效条件", invalidation)}
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


def build_invalidation(tags: list[str], pressure: float, support: float, anchor_high: float, anchor_low: float) -> str:
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


def symbol_summary(symbol: str, name: str, frames: dict[str, pd.DataFrame]) -> str:
    cards = []
    for timeframe in TIMEFRAMES:
        if timeframe in frames:
            cards.append(period_card(symbol, name, timeframe, frames[timeframe]))
    return "\n".join(cards)


def render() -> str:
    data = load_market_data(DATA_FILES)
    blocks = []
    for symbol, meta in SYMBOLS.items():
        daily = data["daily"][data["daily"]["symbol"] == symbol]
        hourly = data["hourly"][data["hourly"]["symbol"] == symbol]
        if daily.empty:
            continue
        frames = build_timeframes(daily, hourly)
        blocks.append(symbol_summary(symbol, meta["name"], frames))
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
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
    }}
    .period-card {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 16px;
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
    .line-item {{
      border-top: 1px solid var(--line);
      padding: 10px 0 0;
      margin-top: 10px;
    }}
    .line-item b {{ color: var(--blue); }}
    .line-item p {{ margin: 6px 0 0; color: #d7dee8; }}
    .level-0 {{ border-left: 4px solid #7b8491; }}
    .level-1 {{ border-left: 4px solid #4cc9f0; }}
    .level-2 {{ border-left: 4px solid #f2c94c; }}
    .level-3 {{ border-left: 4px solid #ff7b72; }}
  </style>
</head>
<body>
  <header>
    <h1>四维分析系统初版</h1>
    <p class="note">减法版：每个品种、每个周期只输出当前状态、当前观察位、等待原因、准备等级、失效条件。</p>
  </header>
  <main>
    {"".join(blocks)}
  </main>
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
