# ============================================================
#  回测引擎
#  对每个品种独立回测，汇总组合绩效
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from config import RISK_PARAMS, BACKTEST_PARAMS
from risk_manager import RiskManager
from signal_generator import generate_signals, Signal


@dataclass
class Trade:
    symbol:    str
    direction: str
    entry_date: pd.Timestamp
    exit_date:  pd.Timestamp
    entry:     float
    exit:      float
    pnl:       float
    net_pnl:   float
    return_pct: float


@dataclass
class BacktestResult:
    symbol:      str
    trades:      list[Trade]
    equity_curve: pd.Series      # date → 资金
    metrics:     dict


# ─────────────────────────────────────────────
#  单品种回测
# ─────────────────────────────────────────────

def backtest_symbol(
    symbol:    str,
    df:        pd.DataFrame,
    capital:   float = RISK_PARAMS["capital"],
    start_date: str  = BACKTEST_PARAMS["start_date"],
    end_date:   str  = BACKTEST_PARAMS["end_date"],
) -> BacktestResult:
    """对单个品种执行回测。"""

    # 过滤回测区间
    df = df.copy()
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)
    if len(df) < 60:
        return BacktestResult(symbol, [], pd.Series(dtype=float), {})

    rm     = RiskManager(capital)
    signals: list[Signal] = generate_signals(symbol, df)

    trades: list[Trade] = []
    pending_open: Signal | None = None

    equity_records: list[tuple[pd.Timestamp, float]] = []

    signal_map: dict[pd.Timestamp, Signal] = {s.date: s for s in signals}

    for _, row in df.iterrows():
        date = row["date"]

        if date in signal_map:
            sig = signal_map[date]

            if sig.action in ("BUY", "SHORT") and rm.can_open(symbol):
                direction = "LONG" if sig.action == "BUY" else "SHORT"
                rm.open_position(symbol, direction, sig.price, sig.atr)
                pending_open = sig

            elif sig.action in ("SELL", "COVER") and symbol in rm.positions:
                result = rm.close_position(symbol, sig.price)
                if result and pending_open:
                    trades.append(Trade(
                        symbol     = symbol,
                        direction  = result["direction"],
                        entry_date = pending_open.date,
                        exit_date  = date,
                        entry      = result["entry"],
                        exit       = result["exit"],
                        pnl        = result["pnl"],
                        net_pnl    = result["net_pnl"],
                        return_pct = result["return_pct"],
                    ))
                    pending_open = None

        equity_records.append((date, rm.equity))

    equity_curve = pd.Series(
        [e for _, e in equity_records],
        index=[d for d, _ in equity_records],
        name=symbol,
    )

    metrics = _compute_metrics(trades, equity_curve, capital)
    return BacktestResult(symbol, trades, equity_curve, metrics)


# ─────────────────────────────────────────────
#  绩效指标计算
# ─────────────────────────────────────────────

def _compute_metrics(
    trades:       list[Trade],
    equity_curve: pd.Series,
    initial_cap:  float,
) -> dict:
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "total_return": 0,
                "max_drawdown": 0, "sharpe": 0, "profit_factor": 0}

    total   = len(trades)
    wins    = sum(1 for t in trades if t.net_pnl > 0)
    gross_p = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_l = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))

    # 最大回撤
    if len(equity_curve) > 1:
        roll_max = equity_curve.cummax()
        dd       = (equity_curve - roll_max) / roll_max
        max_dd   = dd.min()
    else:
        max_dd = 0.0

    # 夏普比（日收益）
    daily_ret = equity_curve.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)

    final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_cap
    total_return = (final_equity - initial_cap) / initial_cap * 100

    return {
        "total_trades":  total,
        "win_trades":    wins,
        "win_rate":      round(wins / total * 100, 1),
        "total_return":  round(total_return, 2),
        "max_drawdown":  round(max_dd * 100, 2),
        "sharpe":        round(sharpe, 2),
        "profit_factor": round(gross_p / gross_l, 2) if gross_l > 0 else float("inf"),
        "avg_pnl":       round(sum(t.net_pnl for t in trades) / total, 2),
    }


# ─────────────────────────────────────────────
#  多品种组合回测
# ─────────────────────────────────────────────

def backtest_portfolio(
    all_data: dict[str, pd.DataFrame],
    symbols:  list[str] | None = None,
    capital:  float = RISK_PARAMS["capital"],
) -> dict:
    """
    对多个品种分别回测，汇总结果。
    返回 {symbol: BacktestResult} 和组合汇总。
    """
    symbols = symbols or list(all_data.keys())
    results: dict[str, BacktestResult] = {}

    for sym in symbols:
        df = all_data.get(sym)
        if df is None or len(df) < 60:
            continue
        results[sym] = backtest_symbol(sym, df, capital=capital)

    # 汇总表
    summary_rows = []
    for sym, res in results.items():
        from config import ALL_SYMBOLS
        m = res.metrics
        summary_rows.append({
            "品种":   ALL_SYMBOLS.get(sym, sym),
            "代码":   sym,
            "交易次数": m.get("total_trades", 0),
            "胜率%":   m.get("win_rate", 0),
            "总收益%":  m.get("total_return", 0),
            "最大回撤%": m.get("max_drawdown", 0),
            "夏普比":   m.get("sharpe", 0),
            "盈亏比":   m.get("profit_factor", 0),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("总收益%", ascending=False)
    return {"results": results, "summary": summary_df}
