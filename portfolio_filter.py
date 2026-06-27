from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from typing import Iterable

import pandas as pd

from config import PORTFOLIO_FILTER_PARAMS


@dataclass(frozen=True)
class SymbolEligibility:
    symbol: str
    allowed: bool
    reason: str
    trades: int
    net_pnl: float
    win_rate: float
    profit_factor: float
    lookback_start: pd.Timestamp | None
    as_of: pd.Timestamp


def _trade_value(trade, name: str):
    return getattr(trade, name, trade.get(name) if isinstance(trade, dict) else None)


def evaluate_symbol_eligibility(
    symbol: str,
    closed_trades: Iterable,
    as_of,
    params: dict | None = None,
) -> SymbolEligibility:
    params = params or PORTFOLIO_FILTER_PARAMS
    as_of_ts = pd.Timestamp(as_of)
    if not params.get("enabled", True):
        return SymbolEligibility(symbol, True, "品种准入关闭", 0, 0.0, 0.0, float("inf"), None, as_of_ts)

    mode = params.get("mode", "rolling_oos")
    if mode == "static_blocklist":
        blocked = symbol in set(params.get("blocked_symbols", []))
        return SymbolEligibility(
            symbol=symbol,
            allowed=not blocked,
            reason="固定禁用品种" if blocked else "固定名单允许",
            trades=0,
            net_pnl=0.0,
            win_rate=0.0,
            profit_factor=float("inf"),
            lookback_start=None,
            as_of=as_of_ts,
        )

    lookback_days = int(params.get("lookback_days", 365))
    min_trades = int(params.get("min_trades", 3))
    min_net_pnl = float(params.get("min_net_pnl", 0))
    min_win_rate = float(params.get("min_win_rate", 35))
    min_profit_factor = float(params.get("min_profit_factor", 1.0))
    lookback_start = as_of_ts - pd.Timedelta(days=lookback_days)

    sample = []
    for trade in closed_trades:
        if _trade_value(trade, "symbol") != symbol:
            continue
        exit_date = pd.Timestamp(_trade_value(trade, "exit_date"))
        if lookback_start <= exit_date < as_of_ts:
            sample.append(trade)

    trades = len(sample)
    if trades < min_trades:
        return SymbolEligibility(
            symbol=symbol,
            allowed=True,
            reason=f"滚动观察期：近{lookback_days}日仅{trades}笔，未达到{min_trades}笔",
            trades=trades,
            net_pnl=round(sum(float(_trade_value(t, "net_pnl") or 0) for t in sample), 2),
            win_rate=0.0,
            profit_factor=float("inf"),
            lookback_start=lookback_start,
            as_of=as_of_ts,
        )

    pnls = [float(_trade_value(t, "net_pnl") or 0) for t in sample]
    net_pnl = sum(pnls)
    win_rate = sum(1 for pnl in pnls if pnl > 0) / trades * 100
    gross_profit = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    failed = []
    if net_pnl < min_net_pnl:
        failed.append(f"净利润{net_pnl:.0f}<{min_net_pnl:.0f}")
    if win_rate < min_win_rate:
        failed.append(f"胜率{win_rate:.1f}%<{min_win_rate:.1f}%")
    if profit_factor < min_profit_factor:
        failed.append(f"盈亏比{profit_factor:.2f}<{min_profit_factor:.2f}")

    allowed = not failed
    pf_text = "inf" if isinf(profit_factor) else f"{profit_factor:.2f}"
    reason = (
        f"滚动准入通过：近{lookback_days}日{trades}笔，净利{net_pnl:.0f}，胜率{win_rate:.1f}%，盈亏比{pf_text}"
        if allowed
        else f"滚动准入暂停：{'; '.join(failed)}"
    )
    return SymbolEligibility(
        symbol=symbol,
        allowed=allowed,
        reason=reason,
        trades=trades,
        net_pnl=round(net_pnl, 2),
        win_rate=round(win_rate, 1),
        profit_factor=round(profit_factor, 2) if not isinf(profit_factor) else float("inf"),
        lookback_start=lookback_start,
        as_of=as_of_ts,
    )


def build_eligibility_snapshot(
    symbols: Iterable[str],
    closed_trades: Iterable,
    as_of,
    params: dict | None = None,
) -> dict[str, SymbolEligibility]:
    return {
        symbol: evaluate_symbol_eligibility(symbol, closed_trades, as_of, params=params)
        for symbol in sorted(set(symbols))
    }
