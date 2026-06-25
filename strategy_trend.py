# ============================================================
#  策略一：趋势跟踪（回调入场）
#  核心逻辑：
#    ① 确认趋势（MA多头/空头排列 + ADX > 20）
#    ② 等待回调（价格向MA20靠近，量能收缩）
#    ③ 触发入场（量能重新放大 + 价格收回MA20上方）
#    ④ 止损 = 回调最低点 - 0.5×ATR
#    ⑤ 止盈 = 2×风险距离（最低盈亏比 1:2）
#
#  过滤条件：
#    - ADX > 20（有趋势才入场）
#    - CI < 61.8（不在震荡市中操作）
#    - K线密度评分 < 60（行情不拥挤）
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from signal_generator import Signal
from config import RISK_PARAMS

STRATEGY_NAME = "趋势跟踪"

# ── 策略参数 ──────────────────────────────────
ADX_MIN        = 20     # 最低趋势强度
CI_MAX         = 61.8   # 震荡指数上限（超过则不操作）
VOL_CONTRACT   = 0.85   # 回调期间量能收缩阈值（< 0.85×均量 = 缩量）
VOL_EXPAND     = 1.10   # 入场触发量能扩张阈值（> 1.1×均量 = 放量）
PULLBACK_BARS  = 8      # 回调等待最大 Bar 数（超过则放弃这次机会）
MA_ALIGN_BARS  = 3      # 均线排列需持续 N 根 K 线
# ─────────────────────────────────────────────


def _ci_at(df: pd.DataFrame, i: int, n: int = 14) -> float:
    """计算第 i 根 K 线处的 Choppiness Index。"""
    if i < n:
        return 50.0
    sub = df.iloc[i - n + 1: i + 1]
    tr = pd.concat([
        sub["high"] - sub["low"],
        (sub["high"] - sub["close"].shift(1)).abs(),
        (sub["low"]  - sub["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    sum_tr = tr.sum()
    hl     = sub["high"].max() - sub["low"].min()
    if hl <= 0 or sum_tr <= 0:
        return 50.0
    return float(np.clip(100 * np.log10(sum_tr / hl) / np.log10(n), 0, 100))


def _is_uptrend(df: pd.DataFrame, i: int) -> bool:
    """判断 i 处是否处于多头趋势（MA5 > MA20 > MA60，ADX足够）。"""
    r = df.iloc[i]
    ma5, ma20, ma60 = r.get("MA5"), r.get("MA20"), r.get("MA60")
    adx = r.get("ADX", 0)
    if any(pd.isna([ma5, ma20, ma60, adx])):
        return False
    return (ma5 > ma20 > ma60) and (adx >= ADX_MIN)


def _is_downtrend(df: pd.DataFrame, i: int) -> bool:
    """判断 i 处是否处于空头趋势。"""
    r = df.iloc[i]
    ma5, ma20, ma60 = r.get("MA5"), r.get("MA20"), r.get("MA60")
    adx = r.get("ADX", 0)
    if any(pd.isna([ma5, ma20, ma60, adx])):
        return False
    return (ma5 < ma20 < ma60) and (adx >= ADX_MIN)


def generate_signals(symbol: str, df: pd.DataFrame) -> list[Signal]:
    """
    趋势跟踪策略逐 Bar 扫描，返回 Signal 列表。

    状态机：
      IDLE  ──趋势确认──►  PULLBACK_WAIT  ──量缩回调──►  ENTRY_READY
            ──触发入场──►  IN_TRADE       ──止损/止盈──►  IDLE
    """
    from indicators import add_all_indicators
    df = add_all_indicators(df).dropna(subset=["MA20", "ATR", "ADX"]).reset_index(drop=True)

    sl_mult = RISK_PARAMS["atr_stop_mult"]

    signals: list[Signal] = []
    state        = "IDLE"   # IDLE / PULLBACK_WAIT / ENTRY_READY / IN_TRADE
    direction    = None     # "LONG" / "SHORT"
    pullback_bar = 0        # 进入回调的 bar index
    pullback_low = 0.0      # 回调期间最低价（做多）
    pullback_high = 0.0     # 回调期间最高价（做空）
    entry_price  = 0.0
    stop_loss    = 0.0
    target       = 0.0

    for i in range(1, len(df)):
        r     = df.iloc[i]
        prev  = df.iloc[i - 1]
        date  = r["date"]
        close = float(r["close"])
        low   = float(r["low"])
        high  = float(r["high"])
        atr   = float(r["ATR"])
        ma20  = float(r["MA20"])
        vol   = float(r.get("volume", 1))
        volma = float(r.get("VOL_MA5", vol))
        ci    = _ci_at(df, i)

        # ══════════ 无仓位阶段 ══════════
        if state == "IDLE":
            if ci >= CI_MAX:
                continue   # 震荡市不找入场

            if _is_uptrend(df, i):
                # 价格开始向 MA20 靠近即进入回调等待
                if close < float(prev["MA20"]) * 1.02:  # 接近或触及 MA20
                    state        = "PULLBACK_WAIT"
                    direction    = "LONG"
                    pullback_bar = i
                    pullback_low = low

            elif _is_downtrend(df, i):
                if close > float(prev["MA20"]) * 0.98:
                    state         = "PULLBACK_WAIT"
                    direction     = "SHORT"
                    pullback_bar  = i
                    pullback_high = high

        # ══════════ 等待量缩回调 ══════════
        elif state == "PULLBACK_WAIT":
            bars_in_pullback = i - pullback_bar

            # 超时放弃
            if bars_in_pullback > PULLBACK_BARS:
                state     = "IDLE"
                direction = None
                continue

            # 趋势已破，放弃
            if direction == "LONG" and not _is_uptrend(df, i):
                state = "IDLE"; continue
            if direction == "SHORT" and not _is_downtrend(df, i):
                state = "IDLE"; continue

            ci = _ci_at(df, i)
            if ci >= CI_MAX:
                state = "IDLE"; continue

            if direction == "LONG":
                pullback_low = min(pullback_low, low)

                # 量缩阶段：等待缩量完成
                vol_contracting = (volma > 0) and (vol < volma * VOL_CONTRACT)

                # 入场触发：价格收回 MA20 上方 + 放量
                vol_expanding = (volma > 0) and (vol > volma * VOL_EXPAND)
                price_recovered = close > ma20

                if price_recovered and vol_expanding:
                    risk      = close - (pullback_low - 0.5 * atr)
                    stop_loss = pullback_low - 0.5 * atr
                    target    = close + max(2 * risk, sl_mult * atr * 1.5)
                    entry_price = close
                    state = "IN_TRADE"
                    signals.append(Signal(
                        date=date, symbol=symbol, action="BUY",
                        price=close, stop_loss=stop_loss, target=target, atr=atr,
                        reason=f"趋势回调入场｜回调低点={pullback_low:.1f}｜CI={ci:.0f}｜量能放大{vol/volma:.1f}x"
                    ))

            elif direction == "SHORT":
                pullback_high = max(pullback_high, high)

                vol_expanding = (volma > 0) and (vol > volma * VOL_EXPAND)
                price_recovered = close < ma20

                if price_recovered and vol_expanding:
                    risk      = (pullback_high + 0.5 * atr) - close
                    stop_loss = pullback_high + 0.5 * atr
                    target    = close - max(2 * risk, sl_mult * atr * 1.5)
                    entry_price = close
                    state = "IN_TRADE"
                    signals.append(Signal(
                        date=date, symbol=symbol, action="SHORT",
                        price=close, stop_loss=stop_loss, target=target, atr=atr,
                        reason=f"趋势反弹做空｜回调高点={pullback_high:.1f}｜CI={ci:.0f}｜量能放大{vol/volma:.1f}x"
                    ))

        # ══════════ 持仓管理 ══════════
        elif state == "IN_TRADE":
            dif = float(r.get("DIF", 0))
            dea = float(r.get("DEA", 0))
            p_dif = float(prev.get("DIF", 0))
            p_dea = float(prev.get("DEA", 0))

            if direction == "LONG":
                exit_reason = ""
                if close <= stop_loss:
                    exit_reason = f"止损({close:.1f}<={stop_loss:.1f})"
                elif close >= target:
                    exit_reason = f"止盈({close:.1f}>={target:.1f})"
                elif dif < dea and p_dif >= p_dea and close < float(r.get("MA5", close)):
                    exit_reason = "MACD死叉+跌破MA5"
                if exit_reason:
                    signals.append(Signal(date, symbol, "SELL", close, 0, 0, atr, exit_reason))
                    state = "IDLE"; direction = None

            elif direction == "SHORT":
                exit_reason = ""
                if close >= stop_loss:
                    exit_reason = f"止损({close:.1f}>={stop_loss:.1f})"
                elif close <= target:
                    exit_reason = f"止盈({close:.1f}<={target:.1f})"
                elif dif > dea and p_dif <= p_dea and close > float(r.get("MA5", close)):
                    exit_reason = "MACD金叉+突破MA5"
                if exit_reason:
                    signals.append(Signal(date, symbol, "COVER", close, 0, 0, atr, exit_reason))
                    state = "IDLE"; direction = None

    return signals
