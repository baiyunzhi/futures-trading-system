# ============================================================
#  Local paper trading engine
#  Generates simulated orders, fills, positions, and equity logs.
# ============================================================

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from config import ALL_SYMBOLS, CONTRACT_SPECS, PAPER_PARAMS, RISK_PARAMS
from signal_generator import Signal


ENTRY_ACTIONS = {"BUY", "SHORT"}
EXIT_ACTIONS = {"SELL", "COVER"}


@dataclass
class PaperPosition:
    symbol: str
    name: str
    direction: str
    entry_date: str
    entry_price: float
    lots: int
    lot_value: float
    stop_loss: float
    target: float
    entry_commission: float
    reason: str


@dataclass
class PaperOrder:
    order_id: int
    date: str
    symbol: str
    name: str
    action: str
    direction: str
    order_type: str
    signal_price: float
    status: str
    reason: str


@dataclass
class PaperFill:
    order_id: int
    date: str
    symbol: str
    name: str
    action: str
    direction: str
    price: float
    lots: int
    pnl: float
    commission: float
    net_pnl: float
    cash_after: float
    reason: str


class PaperTradingEngine:
    def __init__(
        self,
        initial_capital: float = RISK_PARAMS["capital"],
        max_positions: int = RISK_PARAMS["max_positions"],
        max_daily_loss_pct: float = PAPER_PARAMS["max_daily_loss_pct"],
    ):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.max_positions = int(max_positions)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.positions: dict[str, PaperPosition] = {}
        self.pending_entries: dict[str, tuple[Signal, PaperOrder]] = {}
        self.pending_exits: dict[str, tuple[Signal, PaperOrder]] = {}
        self.orders: list[PaperOrder] = []
        self.fills: list[PaperFill] = []
        self.equity_records: list[dict] = []
        self.next_order_id = 1
        self.trading_halted = False

    def run(
        self,
        all_data: dict[str, pd.DataFrame],
        strategy_fn: Callable[[str, pd.DataFrame], list[Signal]],
        symbols: list[str] | None = None,
        start_date: str = PAPER_PARAMS["start_date"],
        end_date: str | None = PAPER_PARAMS["end_date"],
    ) -> dict:
        symbols = symbols or list(all_data.keys())
        data = {
            sym: self._prepare_df(df, start_date, end_date)
            for sym, df in all_data.items()
            if sym in symbols and df is not None and len(df) >= 60
        }
        data = {sym: df for sym, df in data.items() if not df.empty}

        signals = {sym: strategy_fn(sym, df) for sym, df in data.items()}
        signal_map = {
            sym: {pd.Timestamp(sig.date): sig for sig in sigs}
            for sym, sigs in signals.items()
        }
        row_map = {
            sym: {pd.Timestamp(row["date"]): row for _, row in df.iterrows()}
            for sym, df in data.items()
        }
        dates = sorted({d for rows in row_map.values() for d in rows})

        for date in dates:
            prices = {}
            for sym in symbols:
                row = row_map.get(sym, {}).get(date)
                if row is None:
                    continue
                prices[sym] = float(row["close"])
                self._process_pending(sym, date, row)
                self._process_intraday_risk(sym, date, row)

            if not self.trading_halted:
                for sym in symbols:
                    row = row_map.get(sym, {}).get(date)
                    sig = signal_map.get(sym, {}).get(date)
                    if row is not None and sig is not None:
                        self._accept_signal(date, sig)

            equity = self._mark_to_market(prices)
            self.equity_records.append({
                "date": date.strftime("%Y-%m-%d"),
                "cash": round(self.cash, 2),
                "equity": round(equity, 2),
                "open_positions": len(self.positions),
                "halted": self.trading_halted,
            })
            self._check_daily_loss(equity)

        if dates:
            last_date = dates[-1]
            for sym in list(self.positions):
                row = row_map.get(sym, {}).get(last_date)
                if row is not None:
                    self._close_position(sym, last_date, float(row["close"]), "FORCE_CLOSE", "期末强制平仓")

        return self.report()

    def _prepare_df(self, df: pd.DataFrame, start_date: str, end_date: str | None) -> pd.DataFrame:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        out = out[out["date"] >= pd.Timestamp(start_date)]
        if end_date:
            out = out[out["date"] <= pd.Timestamp(end_date)]
        return out.sort_values("date").reset_index(drop=True)

    def _create_order(
        self,
        date: pd.Timestamp,
        symbol: str,
        action: str,
        direction: str,
        order_type: str,
        signal_price: float,
        status: str,
        reason: str,
    ) -> PaperOrder:
        order = PaperOrder(
            order_id=self.next_order_id,
            date=date.strftime("%Y-%m-%d"),
            symbol=symbol,
            name=ALL_SYMBOLS.get(symbol, symbol),
            action=action,
            direction=direction,
            order_type=order_type,
            signal_price=round(float(signal_price), 4),
            status=status,
            reason=reason,
        )
        self.next_order_id += 1
        self.orders.append(order)
        return order

    def _accept_signal(self, date: pd.Timestamp, sig: Signal) -> None:
        if sig.action in ENTRY_ACTIONS:
            direction = "LONG" if sig.action == "BUY" else "SHORT"
            if sig.symbol in self.positions or sig.symbol in self.pending_entries:
                self._create_order(date, sig.symbol, sig.action, direction, "ENTRY", sig.price, "REJECTED", "已有持仓或待开仓")
                return
            if len(self.positions) + len(self.pending_entries) >= self.max_positions:
                self._create_order(date, sig.symbol, sig.action, direction, "ENTRY", sig.price, "REJECTED", "超过最大持仓数")
                return
            order = self._create_order(date, sig.symbol, sig.action, direction, "ENTRY", sig.price, "PENDING", sig.reason)
            self.pending_entries[sig.symbol] = (sig, order)
        elif sig.action in EXIT_ACTIONS and sig.symbol in self.positions:
            direction = self.positions[sig.symbol].direction
            order = self._create_order(date, sig.symbol, sig.action, direction, "EXIT", sig.price, "PENDING", sig.reason)
            self.pending_exits[sig.symbol] = (sig, order)

    def _process_pending(self, symbol: str, date: pd.Timestamp, row: pd.Series) -> None:
        if symbol in self.pending_exits and symbol in self.positions:
            _, order = self.pending_exits.pop(symbol)
            pos = self.positions[symbol]
            price = self._slipped_price(symbol, float(row["open"]), pos.direction, is_entry=False)
            self._close_position(symbol, date, price, order.action, order.reason, order.order_id)

        if symbol in self.pending_entries and symbol not in self.positions:
            sig, order = self.pending_entries.pop(symbol)
            direction = "LONG" if sig.action == "BUY" else "SHORT"
            price = self._slipped_price(symbol, float(row["open"]), direction, is_entry=True)
            self._open_position(date, sig, order, price)

    def _process_intraday_risk(self, symbol: str, date: pd.Timestamp, row: pd.Series) -> None:
        pos = self.positions.get(symbol)
        if pos is None:
            return
        high = float(row["high"])
        low = float(row["low"])
        if pos.direction == "LONG":
            if low <= pos.stop_loss:
                self._close_position(symbol, date, self._slipped_price(symbol, pos.stop_loss, pos.direction, False), "STOP", "日内止损")
            elif high >= pos.target:
                self._close_position(symbol, date, self._slipped_price(symbol, pos.target, pos.direction, False), "TAKE_PROFIT", "日内止盈")
        else:
            if high >= pos.stop_loss:
                self._close_position(symbol, date, self._slipped_price(symbol, pos.stop_loss, pos.direction, False), "STOP", "日内止损")
            elif low <= pos.target:
                self._close_position(symbol, date, self._slipped_price(symbol, pos.target, pos.direction, False), "TAKE_PROFIT", "日内止盈")

    def _open_position(self, date: pd.Timestamp, sig: Signal, order: PaperOrder, price: float) -> None:
        spec = CONTRACT_SPECS.get(sig.symbol, {})
        lot_value = float(spec.get("lot_value", RISK_PARAMS["default_lot_value"]))
        lots = self._calc_lots(price, sig.atr, lot_value)
        if lots <= 0:
            order.status = "REJECTED"
            order.reason = "ATR或资金不足"
            return
        direction = "LONG" if sig.action == "BUY" else "SHORT"
        commission = price * lots * lot_value * RISK_PARAMS["commission_rate"]
        self.cash -= commission
        order.status = "FILLED"
        self.positions[sig.symbol] = PaperPosition(
            symbol=sig.symbol,
            name=ALL_SYMBOLS.get(sig.symbol, sig.symbol),
            direction=direction,
            entry_date=date.strftime("%Y-%m-%d"),
            entry_price=round(price, 4),
            lots=lots,
            lot_value=lot_value,
            stop_loss=round(float(sig.stop_loss), 4),
            target=round(float(sig.target), 4),
            entry_commission=round(commission, 2),
            reason=sig.reason,
        )
        self._record_fill(order.order_id, date, sig.symbol, sig.action, direction, price, lots, 0.0, commission, -commission, sig.reason)

    def _close_position(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float,
        action: str,
        reason: str,
        order_id: int | None = None,
    ) -> None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return
        if order_id is None:
            order = self._create_order(date, symbol, action, pos.direction, "EXIT", price, "FILLED", reason)
            order_id = order.order_id
        else:
            for order in self.orders:
                if order.order_id == order_id:
                    order.status = "FILLED"
                    break
        pnl = (price - pos.entry_price) * pos.lots * pos.lot_value if pos.direction == "LONG" else (pos.entry_price - price) * pos.lots * pos.lot_value
        commission = price * pos.lots * pos.lot_value * RISK_PARAMS["commission_rate"]
        net_pnl = pnl - commission - pos.entry_commission
        self.cash += pnl - commission
        self._record_fill(order_id, date, symbol, action, pos.direction, price, pos.lots, pnl, commission, net_pnl, reason)

    def _record_fill(
        self,
        order_id: int,
        date: pd.Timestamp,
        symbol: str,
        action: str,
        direction: str,
        price: float,
        lots: int,
        pnl: float,
        commission: float,
        net_pnl: float,
        reason: str,
    ) -> None:
        self.fills.append(PaperFill(
            order_id=order_id,
            date=date.strftime("%Y-%m-%d"),
            symbol=symbol,
            name=ALL_SYMBOLS.get(symbol, symbol),
            action=action,
            direction=direction,
            price=round(float(price), 4),
            lots=int(lots),
            pnl=round(float(pnl), 2),
            commission=round(float(commission), 2),
            net_pnl=round(float(net_pnl), 2),
            cash_after=round(float(self.cash), 2),
            reason=reason,
        ))

    def _slipped_price(self, symbol: str, price: float, direction: str, is_entry: bool) -> float:
        spec = CONTRACT_SPECS.get(symbol, {})
        tick_size = float(spec.get("tick_size", RISK_PARAMS["default_tick_size"]))
        slip = RISK_PARAMS["slippage_ticks"] * tick_size
        if direction == "LONG":
            return price + slip if is_entry else price - slip
        return price - slip if is_entry else price + slip

    def _calc_lots(self, price: float, atr: float, lot_value: float) -> int:
        if price <= 0 or atr <= 0 or lot_value <= 0:
            return 0
        risk_amount = self.cash * RISK_PARAMS["max_risk_per_trade"]
        risk_per_lot = atr * RISK_PARAMS["atr_stop_mult"] * lot_value
        return max(0, int(risk_amount / risk_per_lot))

    def _mark_to_market(self, prices: dict[str, float]) -> float:
        equity = self.cash
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos.entry_price)
            if pos.direction == "LONG":
                equity += (price - pos.entry_price) * pos.lots * pos.lot_value
            else:
                equity += (pos.entry_price - price) * pos.lots * pos.lot_value
        return float(equity)

    def _check_daily_loss(self, equity: float) -> None:
        drawdown = (equity - self.initial_capital) / self.initial_capital
        if drawdown <= -self.max_daily_loss_pct:
            self.trading_halted = True

    def report(self) -> dict:
        equity = self.equity_records[-1]["equity"] if self.equity_records else self.cash
        closed_fills = [f for f in self.fills if f.action not in ENTRY_ACTIONS]
        wins = sum(1 for f in closed_fills if f.net_pnl > 0)
        return {
            "account": {
                "initial_capital": round(self.initial_capital, 2),
                "cash": round(self.cash, 2),
                "equity": round(float(equity), 2),
                "return_pct": round((float(equity) - self.initial_capital) / self.initial_capital * 100, 2),
                "open_positions": len(self.positions),
                "orders": len(self.orders),
                "fills": len(self.fills),
                "closed_trades": len(closed_fills),
                "win_rate": round(wins / len(closed_fills) * 100, 1) if closed_fills else 0.0,
                "halted": self.trading_halted,
            },
            "positions": [asdict(p) for p in self.positions.values()],
            "orders": [asdict(o) for o in self.orders],
            "fills": [asdict(f) for f in self.fills],
            "equity_curve": self.equity_records,
        }


def run_paper_session(
    all_data: dict[str, pd.DataFrame],
    strategy_fn: Callable[[str, pd.DataFrame], list[Signal]],
    symbols: list[str] | None = None,
    start_date: str = PAPER_PARAMS["start_date"],
    end_date: str | None = PAPER_PARAMS["end_date"],
    storage_dir: str = PAPER_PARAMS["storage_dir"],
) -> dict:
    engine = PaperTradingEngine()
    report = engine.run(all_data, strategy_fn, symbols=symbols, start_date=start_date, end_date=end_date)
    save_paper_report(report, storage_dir)
    return report


def save_paper_report(report: dict, storage_dir: str = PAPER_PARAMS["storage_dir"]) -> None:
    out = Path(storage_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "account.json").write_text(json.dumps(report["account"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(report["positions"]).to_csv(out / "positions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(report["orders"]).to_csv(out / "orders.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(report["fills"]).to_csv(out / "fills.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(report["equity_curve"]).to_csv(out / "equity_curve.csv", index=False, encoding="utf-8-sig")


def load_paper_report(storage_dir: str = PAPER_PARAMS["storage_dir"]) -> dict:
    path = Path(storage_dir) / "report.json"
    if not path.exists():
        return {"account": {}, "positions": [], "orders": [], "fills": [], "equity_curve": []}
    return json.loads(path.read_text(encoding="utf-8"))


def report_to_frames(report: dict) -> dict[str, pd.DataFrame]:
    return {
        "positions": pd.DataFrame(report.get("positions", [])),
        "orders": pd.DataFrame(report.get("orders", [])),
        "fills": pd.DataFrame(report.get("fills", [])),
        "equity_curve": pd.DataFrame(report.get("equity_curve", [])),
    }
