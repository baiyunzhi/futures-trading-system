from __future__ import annotations

import argparse
import logging

from config import ALL_SYMBOLS, PAPER_PARAMS
from data_fetcher import get_data
from indicators import add_all_indicators
from paper_trading import run_paper_session

import strategy_trend as strategy


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("paper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local paper trading simulation.")
    parser.add_argument("--symbols", nargs="*", default=None, help="Symbols, for example RB0 CU0 M0.")
    parser.add_argument("--start", default=PAPER_PARAMS["start_date"], help="Replay start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=PAPER_PARAMS["end_date"], help="Replay end date, YYYY-MM-DD.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = args.symbols or list(ALL_SYMBOLS.keys())
    logger.info("加载行情数据...")
    all_data = {}
    for sym in symbols:
        df = get_data(sym, use_cache=True)
        if len(df) >= 60:
            all_data[sym] = add_all_indicators(df)
    logger.info("运行本地模拟盘: %s", ", ".join(all_data.keys()))
    report = run_paper_session(
        all_data,
        strategy.generate_signals,
        symbols=list(all_data.keys()),
        start_date=args.start,
        end_date=args.end,
    )
    account = report["account"]
    logger.info(
        "模拟盘完成: equity=%.2f return=%.2f%% closed_trades=%s win_rate=%.1f%%",
        account["equity"],
        account["return_pct"],
        account["closed_trades"],
        account["win_rate"],
    )
    logger.info("输出目录: %s", PAPER_PARAMS["storage_dir"])


if __name__ == "__main__":
    main()
