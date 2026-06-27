from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from data_fetcher import get_all_data
from indicators import add_all_indicators
from portfolio_backtester import PortfolioBacktestResult, backtest_unified_portfolio


PERIODS = [
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-12-31"),
]


def prepare_real_data() -> dict[str, pd.DataFrame]:
    raw = get_all_data(use_cache=True)
    return {symbol: add_all_indicators(df) for symbol, df in raw.items() if df is not None and len(df) >= 80}


def run_period_validation(
    all_data: dict[str, pd.DataFrame],
    periods: list[tuple[str, str, str]] = PERIODS,
) -> tuple[pd.DataFrame, dict[str, PortfolioBacktestResult]]:
    rows = []
    results = {}
    for name, start, end in periods:
        result = backtest_unified_portfolio(all_data, start_date=start, end_date=end)
        results[name] = result
        rows.append({
            "周期": name,
            "开始": start,
            "结束": end,
            "交易次数": result.metrics.get("total_trades", 0),
            "胜率%": result.metrics.get("win_rate", 0),
            "总收益%": result.metrics.get("total_return", 0),
            "最大回撤%": result.metrics.get("max_drawdown", 0),
            "夏普": result.metrics.get("sharpe", 0),
            "盈亏比": result.metrics.get("profit_factor", 0),
        })
    return pd.DataFrame(rows), results


def symbol_contribution(result: PortfolioBacktestResult) -> pd.DataFrame:
    if not result.trades:
        return pd.DataFrame(columns=["品种", "交易次数", "净利润", "胜率%", "平均盈亏"])
    df = pd.DataFrame([asdict(t) for t in result.trades])
    grouped = df.groupby("symbol", as_index=False).agg(
        交易次数=("net_pnl", "count"),
        净利润=("net_pnl", "sum"),
        平均盈亏=("net_pnl", "mean"),
        胜率=("net_pnl", lambda s: (s > 0).mean() * 100),
    )
    grouped = grouped.rename(columns={"symbol": "品种", "胜率": "胜率%"})
    grouped["净利润"] = grouped["净利润"].round(2)
    grouped["平均盈亏"] = grouped["平均盈亏"].round(2)
    grouped["胜率%"] = grouped["胜率%"].round(1)
    return grouped.sort_values("净利润", ascending=False).reset_index(drop=True)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据\n"
    return df.to_markdown(index=False)


def build_validation_report(output: str = "docs/strategy_diagnostics_report.md") -> Path:
    all_data = prepare_real_data()
    period_df, results = run_period_validation(all_data)
    full_result = backtest_unified_portfolio(all_data)
    contrib = symbol_contribution(full_result)

    lines = [
        "# 策略样本外与贡献诊断报告",
        "",
        "更新时间：2026-06-27",
        "",
        "## 分年度统一账户回测",
        "",
        dataframe_to_markdown(period_df),
        "",
        "## 2023-2024 品种贡献",
        "",
        dataframe_to_markdown(contrib),
        "",
        "## 诊断结论",
        "",
        "1. 如果某一年明显亏损，说明系统对市场环境切换不稳定。",
        "2. 如果利润集中在少数品种，说明策略不是全品种稳健优势。",
        "3. 如果盈亏比低于 1.2，优先优化出场和过滤，不优先提高交易频率。",
        "4. 如果最大回撤大于收益，系统不能进入模拟自动执行。",
        "",
    ]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build_validation_report())
