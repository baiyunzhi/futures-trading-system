# ============================================================
#  主程序入口
#  运行: python main.py
#  访问: http://127.0.0.1:8050
# ============================================================

import logging
import pandas as pd
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def main():
    logger.info("=" * 60)
    logger.info("  商品期货交易系统  启动中...")
    logger.info("=" * 60)

    # ── 1. 获取数据 ──
    logger.info("Step 1/5  加载行情数据...")
    from data_fetcher import get_all_data
    all_data = get_all_data(use_cache=True)
    logger.info(f"  加载完成：{len(all_data)} 个品种")

    # ── 2. 计算指标 ──
    logger.info("Step 2/5  计算技术指标...")
    from indicators import add_all_indicators
    all_data_ind = {}
    for sym, df in all_data.items():
        if len(df) >= 60:
            all_data_ind[sym] = add_all_indicators(df)
    logger.info(f"  指标计算完成：{len(all_data_ind)} 个品种")

    # ── 3. 品种评分排名 ──
    logger.info("Step 3/5  品种评分排名...")
    from variety_selector import rank_symbols, get_top_symbols
    rank_df = rank_symbols(all_data_ind)
    top5 = get_top_symbols(rank_df, top_n=5)

    logger.info("  Top 5 品种：")
    for i, sym in enumerate(top5, 1):
        row = rank_df[rank_df["symbol"] == sym].iloc[0]
        from config import ALL_SYMBOLS
        logger.info(f"    {i}. {ALL_SYMBOLS.get(sym, sym):6s}  "
                    f"评分={row['score']:.1f}  方向={row['direction']}")

    # ── 4. 市场状态分析 ──
    logger.info("Step 4/5  市场状态分析...")
    from market_analyzer import analyze_all, results_to_dataframe
    analysis_results = analyze_all(all_data_ind, rank_df)
    state_df = results_to_dataframe(analysis_results)

    logger.info("  各品种状态：")
    for r in analysis_results[:5]:
        from config import ALL_SYMBOLS
        logger.info(f"    {ALL_SYMBOLS.get(r.symbol, r.symbol):6s}  {r.state}  "
                    f"评分={r.score:.1f}  入场={r.entry}  止损={r.stop_loss}  止盈={r.target}")

    # ── 5. 多策略回测 ──
    logger.info("Step 5/5  纯结构策略回测...")
    from backtester import backtest_portfolio
    from portfolio_backtester import backtest_unified_portfolio
    from trade_decision import build_all_trade_decisions
    import strategy_structure as st_structure

    syms = list(all_data_ind.keys())
    bt_trend = backtest_portfolio(all_data_ind, symbols=syms,
                                  strategy_fn=st_structure.generate_signals,
                                  strategy_name="纯结构策略")
    unified_portfolio = backtest_unified_portfolio(all_data_ind)
    trade_decisions = build_all_trade_decisions(all_data_ind, eligibility_snapshot=unified_portfolio.eligibility_snapshot)
    # 兼容 dashboard：portfolio_result 保留"默认"结果（趋势跟踪）
    portfolio_result = bt_trend

    def _log_summary(name, res):
        s = res.get("summary")
        if s is None or s.empty:
            logger.info(f"  [{name}] 无回测结果")
            return
        logger.info(f"  [{name}] Top 5 收益：")
        for _, row in s.head(5).iterrows():
            logger.info(f"    {row['品种']:8s}  收益={row['总收益%']:+.1f}%  "
                        f"胜率={row['胜率%']:.0f}%  夏普={row['夏普比']:.2f}  "
                        f"笔数={row['交易次数']}")

    _log_summary("纯结构策略", bt_trend)
    logger.info(f"  [统一账户] 收益={unified_portfolio.metrics.get('total_return', 0):+.2f}%  "
                f"回撤={unified_portfolio.metrics.get('max_drawdown', 0):.2f}%  "
                f"交易={unified_portfolio.metrics.get('total_trades', 0)}")

    # ── 6. 本地模拟盘 ──
    logger.info("Step 6/6  本地模拟盘回放...")
    from paper_trading import run_paper_session
    paper_report = run_paper_session(
        all_data_ind,
        st_structure.generate_signals,
        symbols=syms,
    )
    pa = paper_report.get("account", {})
    logger.info(f"  模拟盘: 权益={pa.get('equity', 0):,.2f}  "
                f"收益={pa.get('return_pct', 0):+.2f}%  "
                f"平仓={pa.get('closed_trades', 0)}  胜率={pa.get('win_rate', 0):.1f}%")

    # ── 启动仪表板 ──
    logger.info("")
    logger.info("  启动可视化仪表板...")
    logger.info("  ▶  浏览器访问: http://127.0.0.1:8050")
    logger.info("  按 Ctrl+C 退出")
    logger.info("")

    from dashboard import create_app
    app = create_app(
        all_results      = analysis_results,
        rank_df          = rank_df,
        bt_results       = portfolio_result.get("results", {}),
        bt_results_bo    = None,
        bt_results_range = None,
        all_data         = all_data_ind,
        paper_report     = paper_report,
        trade_decisions  = trade_decisions,
        unified_portfolio = unified_portfolio,
    )
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("系统已停止。")
        sys.exit(0)
