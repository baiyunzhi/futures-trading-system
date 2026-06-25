# ============================================================
#  主程序入口
#  运行: python main.py
#  访问: http://127.0.0.1:8050
# ============================================================

import logging
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

    # ── 5. 模拟回测 ──
    logger.info("Step 5/5  模拟回测（历史验证）...")
    from backtester import backtest_portfolio
    portfolio_result = backtest_portfolio(all_data_ind, symbols=list(all_data_ind.keys()))

    summary = portfolio_result.get("summary", None)
    if summary is not None and not summary.empty:
        logger.info("  回测汇总（Top 5 收益）：")
        top_res = summary.head(5)
        for _, row in top_res.iterrows():
            logger.info(f"    {row['品种']:8s}  收益={row['总收益%']:+.1f}%  "
                        f"胜率={row['胜率%']:.0f}%  夏普={row['夏普比']:.2f}")

    # ── 启动仪表板 ──
    logger.info("")
    logger.info("  启动可视化仪表板...")
    logger.info("  ▶  浏览器访问: http://127.0.0.1:8050")
    logger.info("  按 Ctrl+C 退出")
    logger.info("")

    from dashboard import create_app
    app = create_app(
        all_results = analysis_results,
        rank_df     = rank_df,
        bt_results  = portfolio_result.get("results", {}),
        all_data    = all_data_ind,
    )
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("系统已停止。")
        sys.exit(0)
