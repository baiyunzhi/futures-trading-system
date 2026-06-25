# ============================================================
#  回测引擎
#  对每个品种独立回测，汇总组合绩效
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from config import CONTRACT_SPECS, RISK_PARAMS, BACKTEST_PARAMS
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
    lots:      int
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
    symbol:      str,
    df:          pd.DataFrame,
    capital:     float = RISK_PARAMS["capital"],
    start_date:  str   = BACKTEST_PARAMS["start_date"],
    end_date:    str   = BACKTEST_PARAMS["end_date"],
    strategy_fn  = None,   # 可传入自定义策略函数 generate_signals(symbol, df) -> [Signal]
) -> BacktestResult:
    """对单个品种执行回测，支持传入不同策略函数。"""

    # 过滤回测区间
    df = df.copy()
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)
    if len(df) < 60:
        return BacktestResult(symbol, [], pd.Series(dtype=float), {})

    rm = RiskManager(capital)
    fn = strategy_fn or generate_signals
    signals: list[Signal] = fn(symbol, df)

    trades: list[Trade] = []
    open_signal: Signal | None = None
    pending_entry: Signal | None = None
    pending_exit: Signal | None = None

    equity_records: list[tuple[pd.Timestamp, float]] = []

    signal_map: dict[pd.Timestamp, Signal] = {s.date: s for s in signals}
    spec = CONTRACT_SPECS.get(symbol, {})
    lot_value = float(spec.get("lot_value", RISK_PARAMS["default_lot_value"]))
    tick_size = float(spec.get("tick_size", RISK_PARAMS["default_tick_size"]))
    slippage = RISK_PARAMS["slippage_ticks"] * tick_size

    def _entry_price(raw_open: float, direction: str) -> float:
        return raw_open + slippage if direction == "LONG" else raw_open - slippage

    def _exit_price(raw_price: float, direction: str) -> float:
        return raw_price - slippage if direction == "LONG" else raw_price + slippage

    def _record_close(exit_date: pd.Timestamp, price: float) -> None:
        nonlocal open_signal
        result = rm.close_position(symbol, price)
        if result and open_signal:
            trades.append(Trade(
                symbol     = symbol,
                direction  = result["direction"],
                entry_date = open_signal.date,
                exit_date  = exit_date,
                entry      = result["entry"],
                exit       = result["exit"],
                lots       = result["lots"],
                pnl        = result["pnl"],
                net_pnl    = result["net_pnl"],
                return_pct = result["return_pct"],
            ))
            open_signal = None

    def _mark_to_market(close_price: float) -> float:
        pos = rm.positions.get(symbol)
        if not pos:
            return rm.equity
        if pos["direction"] == "LONG":
            unrealized = (close_price - pos["entry"]) * pos["lots"] * pos["lot_value"]
        else:
            unrealized = (pos["entry"] - close_price) * pos["lots"] * pos["lot_value"]
        return rm.equity + unrealized

    for i, row in df.iterrows():
        date = row["date"]
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])

        if pending_exit is not None and symbol in rm.positions:
            direction = rm.positions[symbol]["direction"]
            _record_close(date, _exit_price(open_price, direction))
            pending_exit = None

        if pending_entry is not None and symbol not in rm.positions and rm.can_open(symbol):
            direction = "LONG" if pending_entry.action == "BUY" else "SHORT"
            fill_price = _entry_price(open_price, direction)
            opened = rm.open_position(
                symbol,
                direction,
                fill_price,
                pending_entry.atr,
                lots=None,
                lot_value=lot_value,
                stop_loss=pending_entry.stop_loss,
                target=pending_entry.target,
            )
            open_signal = pending_entry if opened else None
            pending_entry = None

        if symbol in rm.positions:
            pos = rm.positions[symbol]
            direction = pos["direction"]
            stop = float(pos["stop_loss"])
            target = float(pos["target"])

            if direction == "LONG":
                if low_price <= stop:
                    _record_close(date, _exit_price(stop, direction))
                elif high_price >= target:
                    _record_close(date, _exit_price(target, direction))
            else:
                if high_price >= stop:
                    _record_close(date, _exit_price(stop, direction))
                elif low_price <= target:
                    _record_close(date, _exit_price(target, direction))

        if date in signal_map:
            sig = signal_map[date]

            if i < len(df) - 1:
                if sig.action in ("BUY", "SHORT") and rm.can_open(symbol) and symbol not in rm.positions:
                    pending_entry = sig
                elif sig.action in ("SELL", "COVER") and symbol in rm.positions:
                    pending_exit = sig

        equity_records.append((date, _mark_to_market(close_price)))

    if symbol in rm.positions:
        last = df.iloc[-1]
        direction = rm.positions[symbol]["direction"]
        _record_close(last["date"], _exit_price(float(last["close"]), direction))
        if equity_records:
            equity_records[-1] = (last["date"], rm.equity)

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
    def _round(value: float, digits: int = 2) -> float:
        return round(float(value), digits)

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
        "win_rate":      _round(wins / total * 100, 1),
        "total_return":  _round(total_return),
        "max_drawdown":  _round(max_dd * 100),
        "sharpe":        _round(sharpe),
        "profit_factor": _round(gross_p / gross_l) if gross_l > 0 else float("inf"),
        "avg_pnl":       _round(sum(t.net_pnl for t in trades) / total),
    }


# ─────────────────────────────────────────────
#  多品种组合回测
# ─────────────────────────────────────────────

def backtest_portfolio(
    all_data:    dict[str, pd.DataFrame],
    symbols:     list[str] | None = None,
    capital:     float = RISK_PARAMS["capital"],
    strategy_fn  = None,   # 策略函数，None=使用默认
    strategy_name: str = "默认策略",
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
        results[sym] = backtest_symbol(sym, df, capital=capital, strategy_fn=strategy_fn)

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
    return {"results": results, "summary": summary_df, "strategy_name": strategy_name}
