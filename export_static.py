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
import pandas as pd

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
    from market_analyzer import analyze_all, results_to_dataframe, get_detail_text
    from backtester import backtest_portfolio
    from dashboard import build_score_heatmap, build_kline_chart, build_equity_chart, build_paper_equity_chart, build_unified_equity_chart
    from paper_trading import run_paper_session
    from audit_report import audit_items_to_frame, build_audit_items, has_simulated_data
    from portfolio_backtester import backtest_unified_portfolio
    from trade_decision import build_all_trade_decisions, decisions_to_dataframe
    from objective_market_engine import analyze_all_objective_markets, reports_to_dataframe
    from open_interest_sources import summarize_all_open_interest
    import strategy_structure as st_structure

    logger.info("加载数据 + 计算指标...")
    all_data = get_all_data(use_cache=True)
    all_data_ind = {s: add_all_indicators(df) for s, df in all_data.items() if len(df) >= 60}
    logger.info(f"  {len(all_data_ind)} 个品种")

    rank_df = rank_symbols(all_data_ind)
    analysis_results = analyze_all(all_data_ind, rank_df)
    state_df = results_to_dataframe(analysis_results)
    portfolio_result = backtest_portfolio(all_data_ind, symbols=list(all_data_ind.keys()))
    summary_df = portfolio_result.get("summary")
    unified_portfolio = backtest_unified_portfolio(all_data_ind)
    trade_decisions = build_all_trade_decisions(all_data_ind, eligibility_snapshot=unified_portfolio.eligibility_snapshot)
    decision_df = decisions_to_dataframe(trade_decisions)
    objective_reports = analyze_all_objective_markets(all_data_ind)
    objective_df = reports_to_dataframe(objective_reports)
    oi_df = summarize_all_open_interest(all_data_ind).rename(columns={
        "symbol": "品种",
        "available": "是否可用",
        "source": "来源",
        "latest_date": "日期",
        "latest_open_interest": "最新持仓量",
        "change_pct": "变化%",
        "note": "说明",
    })
    paper_report = run_paper_session(all_data_ind, st_structure.generate_signals, symbols=list(all_data_ind.keys()))

    # ── 渲染图表 ──
    logger.info("渲染图表...")
    # AnalysisResult 列表 → symbol 映射（K线图需要传入对应 result）
    result_map = {r.symbol: r for r in analysis_results}

    parts = []
    parts.append(_fig_html(build_score_heatmap(rank_df), "heatmap", include_js=True))
    parts.append(_fig_html(build_equity_chart(portfolio_result.get("results", {})), "equity", include_js=False))
    parts.append(_fig_html(build_paper_equity_chart(paper_report), "paper-equity", include_js=False))
    parts.append(_fig_html(build_unified_equity_chart(unified_portfolio), "unified-equity", include_js=False))

    # 每个品种的K线图 + 行情深度分析文字，预渲染，用 JS 下拉切换显示
    import html as _htmlmod
    kline_divs = []
    detail_divs = []
    symbols = list(all_data_ind.keys())
    for i, sym in enumerate(symbols):
        div = _fig_html(build_kline_chart(all_data_ind[sym], sym, result_map.get(sym)), f"kline-{sym}", include_js=False)
        display = "block" if i == 0 else "none"
        kline_divs.append(f'<div class="kbox" id="box-{sym}" style="display:{display}">{div}</div>')

        # 行情深度分析文字（与 Dash 回调一致，预生成后嵌入静态页）
        try:
            detail_text = get_detail_text(result_map.get(sym)) if result_map.get(sym) else "暂无分析"
        except Exception as e:
            detail_text = f"分析生成失败: {e}"
        detail_divs.append(
            f'<pre class="dbox" id="detail-{sym}" style="display:{display}">{_htmlmod.escape(detail_text)}</pre>'
        )

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
    decision_table = df_to_table(decision_df, "tbl")
    objective_table = df_to_table(objective_df, "tbl")
    oi_table = df_to_table(oi_df, "tbl")
    paper_account = paper_report.get("account", {})
    paper_positions = pd.DataFrame(paper_report.get("positions", []))
    paper_fills = pd.DataFrame(list(reversed(paper_report.get("fills", [])))[:30])
    paper_positions_table = df_to_table(
        paper_positions[[c for c in ["symbol", "name", "direction", "entry_date", "entry_price", "lots", "stop_loss", "target"] if c in paper_positions.columns]],
        "tbl",
    )
    paper_fills_table = df_to_table(
        paper_fills[[c for c in ["date", "symbol", "action", "direction", "price", "lots", "net_pnl", "reason"] if c in paper_fills.columns]],
        "tbl",
    )

    from datetime import datetime
    updated = datetime.now().strftime("%Y-%m-%d %H:%M")
    simulated = has_simulated_data(all_data)
    data_note = "⚠️ 本页基于<strong>仿真数据</strong>生成（akshare 联网失败时的演示模式）。" if simulated \
        else "数据来源：akshare 真实行情。"
    audit_table = df_to_table(
        audit_items_to_frame(build_audit_items(simulated)).rename(columns={
            "level": "级别",
            "status": "状态",
            "title": "问题",
            "detail": "处理说明",
        }),
        "tbl",
    )

    html = TEMPLATE.format(
        heatmap=parts[0],
        equity=parts[1],
        paper_equity=parts[2],
        unified_equity=parts[3],
        unified_return=f"{unified_portfolio.metrics.get('total_return', 0):+.2f}%",
        unified_drawdown=f"{unified_portfolio.metrics.get('max_drawdown', 0):.2f}%",
        unified_trades=unified_portfolio.metrics.get("total_trades", 0),
        unified_win_rate=f"{unified_portfolio.metrics.get('win_rate', 0):.1f}%",
        paper_equity_value=f"{paper_account.get('equity', 0):,.2f}",
        paper_return=f"{paper_account.get('return_pct', 0):+.2f}%",
        paper_positions_count=paper_account.get("open_positions", 0),
        paper_closed_trades=paper_account.get("closed_trades", 0),
        paper_win_rate=f"{paper_account.get('win_rate', 0):.1f}%",
        paper_halted="ON" if paper_account.get("halted") else "OFF",
        paper_positions_table=paper_positions_table,
        paper_fills_table=paper_fills_table,
        kline_boxes="\n".join(kline_divs),
        detail_boxes="\n".join(detail_divs),
        options=options,
        objective_table=objective_table,
        oi_table=oi_table,
        decision_table=decision_table,
        summary_table=summary_table,
        state_table=state_table,
        audit_table=audit_table,
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
  .wide{{overflow-x:auto}}
  footer{{color:#555;font-size:12px;text-align:center;padding:20px}}
  .kline-row{{display:flex;gap:16px;flex-wrap:wrap}}
  .kline-col{{flex:1;min-width:480px}}
  .detail-col{{width:340px;min-width:300px;background:#0e1117;border:1px solid #2a2a3e;border-radius:6px;padding:12px}}
  .dbox{{color:#ccc;font-size:12px;white-space:pre-wrap;word-break:break-word;max-height:680px;overflow-y:auto;margin:0;font-family:'Consolas','SF Mono',monospace;line-height:1.6}}
  .metric-row{{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:10px;margin-bottom:12px}}
  .metric{{background:#0e1117;border:1px solid #2a2a3e;border-radius:6px;padding:10px}}
  .metric span{{display:block;color:#888;font-size:12px}}
  .metric strong{{display:block;color:#e0e0e0;font-size:18px;margin-top:4px}}
  @media(max-width:900px){{.metric-row{{grid-template-columns:repeat(2,minmax(120px,1fr))}}}}
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
    <h2>K 线分析 + 行情深度分析（可切换品种）</h2>
    <div style="margin-bottom:10px">
      <label style="color:#aaa;margin-right:8px">选择品种：</label>
      <select id="symSelect" onchange="switchSym(this.value)">
        {options}
      </select>
    </div>
    <div class="kline-row">
      <div class="kline-col">{kline_boxes}</div>
      <div class="detail-col">
        <h3 style="color:#26a69a;font-size:14px;margin:0 0 8px">📈 行情深度分析</h3>
        {detail_boxes}
      </div>
    </div>
  </div>

  <div class="card">
    <h2>四维客观行情描述与下注计划</h2>
    <div class="wide">{objective_table}</div>
  </div>

  <div class="card">
    <h2>持仓量数据状态</h2>
    <div class="wide">{oi_table}</div>
  </div>

  <div class="card">
    <h2>可执行交易决策</h2>
    <div class="wide">{decision_table}</div>
  </div>

  <div class="card">
    <h2>统一账户组合回测</h2>
    <div class="metric-row">
      <div class="metric"><span>收益率</span><strong>{unified_return}</strong></div>
      <div class="metric"><span>最大回撤</span><strong>{unified_drawdown}</strong></div>
      <div class="metric"><span>交易次数</span><strong>{unified_trades}</strong></div>
      <div class="metric"><span>胜率</span><strong>{unified_win_rate}</strong></div>
    </div>
    {unified_equity}
  </div>

  <div class="card">
    <h2>各品种回测净值曲线</h2>
    {equity}
  </div>

  <div class="card">
    <h2>本地模拟盘</h2>
    <div class="metric-row">
      <div class="metric"><span>权益</span><strong>{paper_equity_value}</strong></div>
      <div class="metric"><span>收益率</span><strong>{paper_return}</strong></div>
      <div class="metric"><span>持仓</span><strong>{paper_positions_count}</strong></div>
      <div class="metric"><span>平仓笔数</span><strong>{paper_closed_trades}</strong></div>
      <div class="metric"><span>胜率</span><strong>{paper_win_rate}</strong></div>
      <div class="metric"><span>熔断</span><strong>{paper_halted}</strong></div>
    </div>
    {paper_equity}
    <h2>当前持仓</h2>
    {paper_positions_table}
    <h2 style="margin-top:14px">最近成交</h2>
    {paper_fills_table}
  </div>

  <div class="card">
    <h2>系统审计与漏洞修复状态</h2>
    {audit_table}
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
  document.querySelectorAll('.dbox').forEach(function(b){{b.style.display='none';}});
  var box=document.getElementById('box-'+sym);
  if(box){{box.style.display='block'; window.dispatchEvent(new Event('resize'));}}
  var det=document.getElementById('detail-'+sym);
  if(det){{det.style.display='block';}}
}}
</script>
</body>
</html>"""


if __name__ == "__main__":
    build()
