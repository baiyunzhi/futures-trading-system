# ============================================================
#  静态页面导出
#  把整套分析结果渲染成一个自包含 HTML（docs/index.html），
#  用于发布到 GitHub Pages，无需后端即可在线打开。
#  运行: python export_static.py
# ============================================================

from __future__ import annotations
import logging
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("export")

DOCS = Path(__file__).resolve().parent / "docs"
DOCS.mkdir(exist_ok=True)

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def _fig_html(fig: go.Figure, div_id: str, include_js: bool) -> str:
    """把单个 figure 渲染成 div 片段；第一个图内联 plotly.js，其余复用。"""
    return pio.to_html(
        fig,
        include_plotlyjs=("cdn" if include_js else False),
        full_html=False,
        div_id=div_id,
        config=PLOT_CONFIG,
    )


def build():
    from config import ALL_SYMBOLS
    from data_fetcher import get_all_data
    from indicators import add_all_indicators
    from variety_selector import rank_symbols
    from market_analyzer import analyze_all, results_to_dataframe
    from backtester import backtest_portfolio
    from dashboard import build_score_heatmap, build_kline_chart, build_equity_chart

    logger.info("加载数据 + 计算指标...")
    all_data = get_all_data(use_cache=True)
    all_data_ind = {s: add_all_indicators(df) for s, df in all_data.items() if len(df) >= 60}
    logger.info(f"  {len(all_data_ind)} 个品种")

    rank_df = rank_symbols(all_data_ind)
    analysis_results = analyze_all(all_data_ind, rank_df)
    state_df = results_to_dataframe(analysis_results)
    portfolio_result = backtest_portfolio(all_data_ind, symbols=list(all_data_ind.keys()))
    summary_df = portfolio_result.get("summary")

    # ── 渲染图表 ──
    logger.info("渲染图表...")
    # AnalysisResult 列表 → symbol 映射（K线图需要传入对应 result）
    result_map = {r.symbol: r for r in analysis_results}

    parts = []
    parts.append(_fig_html(build_score_heatmap(rank_df), "heatmap", include_js=True))
    parts.append(_fig_html(build_equity_chart(portfolio_result.get("results", {})), "equity", include_js=False))

    # 每个品种的K线图预渲染，用 JS 下拉切换显示
    kline_divs = []
    symbols = list(all_data_ind.keys())
    for i, sym in enumerate(symbols):
        div = _fig_html(build_kline_chart(all_data_ind[sym], sym, result_map.get(sym)), f"kline-{sym}", include_js=False)
        display = "block" if i == 0 else "none"
        kline_divs.append(f'<div class="kbox" id="box-{sym}" style="display:{display}">{div}</div>')

    options = "\n".join(
        f'<option value="{s}">{ALL_SYMBOLS.get(s, s)} ({s})</option>' for s in symbols
    )

    # ── 表格 HTML ──
    def df_to_table(df, cls):
        if df is None or df.empty:
            return "<p style='color:#888'>无数据</p>"
        head = "".join(f"<th>{c}</th>" for c in df.columns)
        rows = ""
        for _, r in df.iterrows():
            tds = "".join(f"<td>{r[c]}</td>" for c in df.columns)
            rows += f"<tr>{tds}</tr>"
        return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'

    summary_table = df_to_table(summary_df, "tbl")
    state_table = df_to_table(state_df, "tbl")

    from datetime import datetime
    updated = datetime.now().strftime("%Y-%m-%d %H:%M")
    simulated = any(df.get("is_simulated", False).any() if "is_simulated" in df else False
                    for df in all_data.values())
    data_note = "⚠️ 本页基于<strong>仿真数据</strong>生成（akshare 联网失败时的演示模式）。" if simulated \
        else "数据来源：akshare 真实行情。"

    html = TEMPLATE.format(
        heatmap=parts[0],
        equity=parts[1],
        kline_boxes="\n".join(kline_divs),
        options=options,
        summary_table=summary_table,
        state_table=state_table,
        updated=updated,
        data_note=data_note,
    )

    out = DOCS / "index.html"
    out.write_text(html, encoding="utf-8")
    logger.info(f"已导出: {out}  ({out.stat().st_size//1024} KB)")


TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>商品期货交易系统 · 在线仪表板</title>
<style>
  body{{background:#0e1117;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:0 16px 40px}}
  h1{{font-size:20px;padding:18px 0 4px;color:#fff}}
  .note{{color:#888;font-size:13px;margin-bottom:16px}}
  .card{{background:#161b27;border:1px solid #2a2a3e;border-radius:8px;padding:14px;margin-bottom:16px}}
  .grid{{display:flex;gap:16px;flex-wrap:wrap}}
  .grid .card{{flex:1;min-width:340px}}
  h2{{font-size:15px;color:#e0e0e0;margin:0 0 10px}}
  select{{background:#0e1117;color:#e0e0e0;border:1px solid #2a2a3e;border-radius:6px;padding:6px 10px;font-size:13px}}
  table.tbl{{width:100%;border-collapse:collapse;font-size:12px}}
  table.tbl th{{background:#0e1117;color:#aaa;padding:6px 8px;border:1px solid #2a2a3e;text-align:center}}
  table.tbl td{{padding:5px 8px;border:1px solid #2a2a3e;text-align:center;color:#ddd}}
  footer{{color:#555;font-size:12px;text-align:center;padding:20px}}
</style>
</head>
<body>
  <h1>🔭 商品期货交易系统 · 在线仪表板</h1>
  <div class="note">最后更新：{updated} ｜ {data_note} 仅供学习研究，不构成投资建议。</div>

  <div class="card">
    <h2>品种综合评分排行</h2>
    {heatmap}
  </div>

  <div class="card">
    <h2>K 线分析（可切换品种）</h2>
    <div style="margin-bottom:10px">
      <label style="color:#aaa;margin-right:8px">选择品种：</label>
      <select id="symSelect" onchange="switchSym(this.value)">
        {options}
      </select>
    </div>
    {kline_boxes}
  </div>

  <div class="card">
    <h2>各品种回测净值曲线</h2>
    {equity}
  </div>

  <div class="grid">
    <div class="card">
      <h2>📊 回测绩效汇总</h2>
      {summary_table}
    </div>
    <div class="card">
      <h2>🎯 当前市场状态</h2>
      {state_table}
    </div>
  </div>

  <footer>⚠️ 历史回测不代表未来表现，商品期货风险较大，请严格执行止损。</footer>

<script>
function switchSym(sym){{
  document.querySelectorAll('.kbox').forEach(function(b){{b.style.display='none';}});
  var box=document.getElementById('box-'+sym);
  if(box){{box.style.display='block'; window.dispatchEvent(new Event('resize'));}}
}}
</script>
</body>
</html>"""


if __name__ == "__main__":
    build()
