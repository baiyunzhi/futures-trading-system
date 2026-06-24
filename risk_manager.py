# ============================================================
#  风险管理模块
#  计算开仓手数、校验仓位限制、动态跟踪资金变化
# ============================================================

from __future__ import annotations
import numpy as np
from config import RISK_PARAMS


class RiskManager:
    """
    基于 ATR 的动态止损仓位管理器。

    核心思路：
        每笔交易亏损上限 = 总资金 × max_risk_per_trade
        开仓手数 = 亏损上限 / (ATR × atr_stop_mult × 每手乘数)
    """

    def __init__(self, capital: float = RISK_PARAMS["capital"]):
        self.initial_capital  = capital
        self.capital          = capital
        self.positions: dict[str, dict] = {}   # symbol → position info
        self.max_positions    = RISK_PARAMS["max_positions"]
        self.max_risk         = RISK_PARAMS["max_risk_per_trade"]
        self.sl_mult          = RISK_PARAMS["atr_stop_mult"]
        self.tp_mult          = RISK_PARAMS["atr_target_mult"]
        self.commission_rate  = RISK_PARAMS["commission_rate"]

    # ── 仓位计算 ──────────────────────────────

    def calc_lots(
        self,
        price:       float,
        atr:         float,
        lot_value:   float = 1.0,   # 每手价值（元/点）
    ) -> int:
        """
        计算合理开仓手数。
        lot_value 默认 1 表示忽略合约乘数（仅用价差衡量风险）。
        实际使用时传入合约乘数（如螺纹钢 10 元/吨/手）。
        """
        risk_amount = self.capital * self.max_risk
        atr_risk    = atr * self.sl_mult * lot_value
        if atr_risk <= 0:
            return 1
        lots = int(risk_amount / atr_risk)
        return max(1, lots)

    # ── 仓位校验 ──────────────────────────────

    def can_open(self, symbol: str) -> bool:
        """是否允许新开仓。"""
        if symbol in self.positions:
            return False   # 已有持仓，不重复开
        return len(self.positions) < self.max_positions

    # ── 开仓 / 平仓记录 ───────────────────────

    def open_position(
        self,
        symbol:     str,
        direction:  str,   # "LONG" / "SHORT"
        price:      float,
        atr:        float,
        lots:       int   = 1,
        lot_value:  float = 1.0,
    ) -> dict:
        """记录开仓，返回仓位信息。"""
        sl_dist = atr * self.sl_mult
        tp_dist = atr * self.tp_mult

        if direction == "LONG":
            stop_loss = price - sl_dist
            target    = price + tp_dist
        else:
            stop_loss = price + sl_dist
            target    = price - tp_dist

        # 手续费
        commission = price * lots * lot_value * self.commission_rate
        self.capital -= commission

        pos = {
            "symbol":    symbol,
            "direction": direction,
            "entry":     price,
            "stop_loss": round(stop_loss, 1),
            "target":    round(target, 1),
            "lots":      lots,
            "lot_value": lot_value,
            "atr":       atr,
            "commission": commission,
        }
        self.positions[symbol] = pos
        return pos

    def close_position(
        self,
        symbol: str,
        price:  float,
    ) -> dict:
        """
        平仓，更新资金，返回交易结果。
        """
        if symbol not in self.positions:
            return {}

        pos = self.positions.pop(symbol)
        lots      = pos["lots"]
        lot_value = pos["lot_value"]
        entry     = pos["entry"]
        direction = pos["direction"]

        if direction == "LONG":
            pnl = (price - entry) * lots * lot_value
        else:
            pnl = (entry - price) * lots * lot_value

        commission = price * lots * lot_value * self.commission_rate
        net_pnl    = pnl - commission
        self.capital += net_pnl

        return {
            "symbol":    symbol,
            "direction": direction,
            "entry":     entry,
            "exit":      price,
            "lots":      lots,
            "pnl":       round(pnl, 2),
            "commission": round(commission + pos["commission"], 2),
            "net_pnl":   round(net_pnl, 2),
            "return_pct": round(net_pnl / self.initial_capital * 100, 3),
        }

    # ── 状态查询 ──────────────────────────────

    @property
    def equity(self) -> float:
        return self.capital

    @property
    def return_pct(self) -> float:
        return (self.capital - self.initial_capital) / self.initial_capital * 100

    def summary(self) -> dict:
        return {
            "equity":       round(self.equity, 2),
            "return_pct":   round(self.return_pct, 2),
            "open_positions": len(self.positions),
            "positions":    list(self.positions.keys()),
        }
