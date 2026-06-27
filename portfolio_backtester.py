from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import RISK_PARAMS, SYMBOL_SECTOR
from config import CONTRACT_SPECS
from portfolio_filter import SymbolEligibility, build_eligibility_snapshot, evaluate_symbol_eligibility
from signal_generator import Signal
from trade_decision import build_trade_decision, calc_risk_first_lots, strategy_for_regime


@dataclass
class PortfolioTrade:
    symbol: str
    direction: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry: float
    exit: float
    lots: int
    pnl: float
    commission: float
    net_pnl: float
    reason: str


@dataclass
class PortfolioBacktestResult:
    trades: list[PortfolioTrade]
    equity_curve: pd.Series
    metrics: dict
    positions: dict
    eligibility_snapshot: dict[str, SymbolEligibility] = field(default_factory=dict)


def _spec(symbol: str) -> tuple[float, float]:
    spec = CONTRACT_SPECS.get(symbol, {})
    return (
        float(spec.get("lot_value", RISK_PARAMS["default_lot_value"])),
        float(spec.get("tick_size", RISK_PARAMS["default_tick_size"])),
    )


def _slipped(symbol: str, price: float, direction: str, is_entry: bool) -> float:
    _, tick_size = _spec(symbol)
    slip = RISK_PARAMS["slippage_ticks"] * tick_size
    if direction == "LONG":
        return price + slip if is_entry else price - slip
    return price - slip if is_entry else price + slip


def _adjust_brackets(sig: Signal, fill_price: float, direction: str) -> tuple[float, float]:
    if direction == "LONG":
        stop_dist = max(float(sig.price) - float(sig.stop_loss), float(sig.atr) * RISK_PARAMS["atr_stop_mult"])
        target_dist = max(float(sig.target) - float(sig.price), float(sig.atr) * RISK_PARAMS["atr_target_mult"])
        return fill_price - stop_dist, fill_price + target_dist
    stop_dist = max(float(sig.stop_loss) - float(sig.price), float(sig.atr) * RISK_PARAMS["atr_stop_mult"])
    target_dist = max(float(sig.price) - float(sig.target), float(sig.atr) * RISK_PARAMS["atr_target_mult"])
    return fill_price + stop_dist, fill_price - target_dist


def _metrics(trades: list[PortfolioTrade], equity_curve: pd.Series, initial_capital: float) -> dict:
    if equity_curve.empty:
        return {"total_trades": 0, "total_return": 0, "max_drawdown": 0, "sharpe": 0, "win_rate": 0, "profit_factor": 0}
    closed = len(trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    gross_p = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_l = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
    roll_max = equity_curve.cummax()
    dd = (equity_curve - roll_max) / roll_max
    ret = equity_curve.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if len(ret) > 1 and ret.std() > 0 else 0
    return {
        "total_trades": closed,
        "win_trades": wins,
        "win_rate": round(wins / closed * 100, 1) if closed else 0,
        "total_return": round((float(equity_curve.iloc[-1]) - initial_capital) / initial_capital * 100, 2),
        "max_drawdown": round(float(dd.min()) * 100, 2),
        "sharpe": round(float(sharpe), 2),
        "profit_factor": round(gross_p / gross_l, 2) if gross_l > 0 else float("inf"),
        "avg_pnl": round(sum(t.net_pnl for t in trades) / closed, 2) if closed else 0,
    }


def backtest_unified_portfolio(
    all_data: dict[str, pd.DataFrame],
    capital: float = RISK_PARAMS["capital"],
    max_positions: int = RISK_PARAMS["max_positions"],
    max_per_sector: int = 1,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    blocked_symbols: list[str] | None = None,
) -> PortfolioBacktestResult:
    data = {}
    for symbol, df in all_data.items():
        if blocked_symbols is not None and symbol in set(blocked_symbols):
            continue
        if df is None or len(df) < 80:
            continue
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        out = out[(out["date"] >= pd.Timestamp(start_date)) & (out["date"] <= pd.Timestamp(end_date))]
        if len(out) >= 80:
            data[symbol] = out.sort_values("date").reset_index(drop=True)

    row_map = {sym: {pd.Timestamp(row["date"]): row for _, row in df.iterrows()} for sym, df in data.items()}
    dates = sorted({d for rows in row_map.values() for d in rows})
    cash = float(capital)
    positions: dict[str, dict] = {}
    pending_entries: dict[str, Signal] = {}
    pending_exits: dict[str, Signal] = {}
    trades: list[PortfolioTrade] = []
    equity_records: list[tuple[pd.Timestamp, float]] = []
    signal_map: dict[tuple[str, pd.Timestamp], list[Signal]] = {}
    signal_regime_map: dict[tuple[str, pd.Timestamp], set[str]] = {}
    consecutive_losses = 0
    cooldown_days = 0
    peak_equity = float(capital)

    for symbol, df in data.items():
        for regime in ("TREND", "BREAKOUT", "RANGE"):
            strategy_fn = strategy_for_regime(regime)
            if strategy_fn is None:
                continue
            try:
                sigs = strategy_fn(symbol, df)
            except Exception:
                sigs = []
            for sig in sigs:
                date_i = pd.Timestamp(sig.date)
                signal_map.setdefault((symbol, date_i), []).append(sig)
                regimes = {"TREND", "BREAKOUT", "RANGE"} if getattr(strategy_fn, "__module__", "") == "strategy_structure" else {regime}
                signal_regime_map.setdefault((symbol, date_i), set()).update(regimes)

    def close_position(symbol: str, date: pd.Timestamp, price: float, reason: str) -> None:
        nonlocal cash, consecutive_losses
        pos = positions.pop(symbol, None)
        if not pos:
            return
        lots = pos["lots"]
        lot_value = pos["lot_value"]
        direction = pos["direction"]
        pnl = (price - pos["entry"]) * lots * lot_value if direction == "LONG" else (pos["entry"] - price) * lots * lot_value
        commission = price * lots * lot_value * RISK_PARAMS["commission_rate"]
        net_pnl = pnl - commission - pos["entry_commission"]
        cash += pnl - commission
        consecutive_losses = consecutive_losses + 1 if net_pnl < 0 else 0
        trades.append(PortfolioTrade(
            symbol=symbol,
            direction=direction,
            entry_date=pos["entry_date"],
            exit_date=date,
            entry=round(pos["entry"], 4),
            exit=round(price, 4),
            lots=lots,
            pnl=round(pnl, 2),
            commission=round(commission + pos["entry_commission"], 2),
            net_pnl=round(net_pnl, 2),
            reason=reason,
        ))

    def mark_to_market(prices: dict[str, float]) -> float:
        equity = cash
        for sym, pos in positions.items():
            price = prices.get(sym, pos["entry"])
            if pos["direction"] == "LONG":
                equity += (price - pos["entry"]) * pos["lots"] * pos["lot_value"]
            else:
                equity += (pos["entry"] - price) * pos["lots"] * pos["lot_value"]
        return float(equity)

    def sector_open_count(sector: str) -> int:
        return sum(1 for sym in positions if SYMBOL_SECTOR.get(sym, "") == sector)

    for date in dates:
        if cooldown_days > 0:
            cooldown_days -= 1
        prices = {}
        for symbol, rows in row_map.items():
            row = rows.get(date)
            if row is None:
                continue
            open_price = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            prices[symbol] = close

            if symbol in pending_exits and symbol in positions:
                direction = positions[symbol]["direction"]
                fill = _slipped(symbol, open_price, direction, is_entry=False)
                reason = pending_exits.pop(symbol).reason
                close_position(symbol, date, fill, reason)

            if symbol in pending_entries and symbol not in positions:
                sig = pending_entries.pop(symbol)
                eligibility = evaluate_symbol_eligibility(symbol, trades, date)
                if not eligibility.allowed:
                    continue
                direction = "LONG" if sig.action == "BUY" else "SHORT"
                sector = SYMBOL_SECTOR.get(symbol, "")
                if len(positions) < max_positions and sector_open_count(sector) < max_per_sector:
                    fill = _slipped(symbol, open_price, direction, is_entry=True)
                    stop, target = _adjust_brackets(sig, fill, direction)
                    lot_value, _ = _spec(symbol)
                    risk_scale = 1.0
                    if consecutive_losses >= 3:
                        risk_scale = 0.25
                    elif consecutive_losses >= 2:
                        risk_scale = 0.5
                    risk_pct = RISK_PARAMS["max_risk_per_trade"] * risk_scale
                    risk_amount, lots = calc_risk_first_lots(mark_to_market(prices), fill, stop, lot_value, risk_pct=risk_pct)
                    if lots > 0:
                        commission = fill * lots * lot_value * RISK_PARAMS["commission_rate"]
                        cash -= commission
                        positions[symbol] = {
                            "direction": direction,
                            "entry_date": date,
                            "entry": fill,
                            "stop_loss": stop,
                            "target": target,
                            "lots": lots,
                            "lot_value": lot_value,
                            "entry_commission": commission,
                            "risk_amount": risk_amount,
                        }

            if symbol in positions:
                pos = positions[symbol]
                direction = pos["direction"]
                stop = float(pos["stop_loss"])
                target = float(pos["target"])
                if direction == "LONG":
                    if low <= stop:
                        close_position(symbol, date, _slipped(symbol, stop, direction, False), "组合回测止损")
                    elif high >= target:
                        close_position(symbol, date, _slipped(symbol, target, direction, False), "组合回测止盈")
                else:
                    if high >= stop:
                        close_position(symbol, date, _slipped(symbol, stop, direction, False), "组合回测止损")
                    elif low <= target:
                        close_position(symbol, date, _slipped(symbol, target, direction, False), "组合回测止盈")

        for symbol, df in data.items():
            row = row_map[symbol].get(date)
            if row is None:
                continue
            history = df[df["date"] <= date].reset_index(drop=True)
            if len(history) < 80:
                continue
            try:
                todays = signal_map.get((symbol, date), [])
                if not todays:
                    continue
                history = df[df["date"] <= date].reset_index(drop=True)
                if len(history) < 80:
                    continue
                eligibility = evaluate_symbol_eligibility(symbol, trades, date)
                decision = build_trade_decision(symbol, history, mark_to_market(prices), eligibility=eligibility)
                if decision.regime not in signal_regime_map.get((symbol, date), set()):
                    continue
                sig = todays[-1]
                if sig.action in ("BUY", "SHORT") and symbol not in positions and decision.tradable and cooldown_days <= 0:
                    pending_entries[symbol] = sig
                elif sig.action in ("SELL", "COVER") and symbol in positions:
                    pending_exits[symbol] = sig
            except Exception:
                continue

        equity = mark_to_market(prices)
        peak_equity = max(peak_equity, equity)
        drawdown_from_peak = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        if drawdown_from_peak <= -0.06 and cooldown_days <= 0:
            cooldown_days = 20
        equity_records.append((date, equity))

    if dates:
        last_date = dates[-1]
        for symbol in list(positions):
            row = row_map.get(symbol, {}).get(last_date)
            if row is not None:
                direction = positions[symbol]["direction"]
                close_position(symbol, last_date, _slipped(symbol, float(row["close"]), direction, False), "期末强制平仓")
        if equity_records:
            equity_records[-1] = (last_date, cash)

    equity_curve = pd.Series([v for _, v in equity_records], index=[d for d, _ in equity_records], name="unified_portfolio")
    snapshot_date = dates[-1] if dates else pd.Timestamp(end_date)
    return PortfolioBacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        metrics=_metrics(trades, equity_curve, capital),
        positions=positions,
        eligibility_snapshot=build_eligibility_snapshot(data.keys(), trades, snapshot_date),
    )


def portfolio_trades_frame(trades: list[PortfolioTrade]) -> pd.DataFrame:
    return pd.DataFrame([t.__dict__ for t in trades])
