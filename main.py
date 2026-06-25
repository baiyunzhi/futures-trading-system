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
    logger.info("Step 5/5  多策略回测（两套策略对比）...")
    from backtester import backtest_portfolio
    import strategy_trend as st_trend
    import strategy_breakout as st_bo
    import strategy_range as st_range

    syms = list(all_data_ind.keys())
    bt_trend = backtest_portfolio(all_data_ind, symbols=syms,
                                  strategy_fn=st_trend.generate_signals,
                                  strategy_name="趋势跟踪")
    bt_bo    = backtest_portfolio(all_data_ind, symbols=syms,
                                  strategy_fn=st_bo.generate_signals,
                                  strategy_name="突破追涨")
    bt_range = backtest_portfolio(all_data_ind, symbols=syms,
                                  strategy_fn=st_range.generate_signals,
                                  strategy_name="区间高抛低吸")
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

    _log_summary("趋势跟踪", bt_trend)
    _log_summary("突破追涨", bt_bo)
    _log_summary("区间高抛低吸", bt_range)

    # 合并对比
    s1 = bt_trend.get("summary", pd.DataFrame()).add_suffix("_趋势")
    s2 = bt_bo.get("summary",    pd.DataFrame()).add_suffix("_突破")
    if not s1.empty and not s2.empty:
        merged = s1.rename(columns={"品种_趋势": "品种", "代码_趋势": "代码"}).merge(
                 s2.rename(columns={"品种_突破": "品种", "代码_突破": "代码"}),
                 on=["品种","代码"], how="outer").fillna(0)
        logger.info("  ── 两套策略对比（按趋势策略收益排序）──")
        for _, row in merged.sort_values("总收益%_趋势", ascending=False).head(8).iterrows():
            logger.info(f"    {row['品种']:8s}  趋势={row['总收益%_趋势']:+.1f}%  突破={row['总收益%_突破']:+.1f}%")

    # ── 6. 本地模拟盘 ──
    logger.info("Step 6/6  本地模拟盘回放...")
    from paper_trading import run_paper_session
    paper_report = run_paper_session(
        all_data_ind,
        st_trend.generate_signals,
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
        bt_results_bo    = bt_bo.get("results", {}),
        bt_results_range = bt_range.get("results", {}),
        all_data         = all_data_ind,
        paper_report     = paper_report,
    )
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("系统已停止。")
        sys.exit(0)
