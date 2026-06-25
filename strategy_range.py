# ============================================================
#  策略三：区间高抛低吸（均值回归）
#  核心逻辑：
#    ① CI > 55，确认处于震荡区间（不在趋势市操作）
#    ② 识别区间边界：N日滚动高点=阻力，N日滚动低点=支撑
#    ③ 价格触及边界 + 量能萎缩（背离）+ RSI极值 → 对打入场
#    ④ 止损：区间边界外 1×ATR（绝不让错误持续）
#    ⑤ 目标：区间中轴（只取一半利润，不贪心）
#
#  核心约束（区别于前两套策略）：
#    - 只在 CI > 55 的市场中使用
#    - 信号质量要求高：量能背离 + RSI极值必须同时满足
#    - 仓位标记 RANGE_TRADE=True，供风控模块缩小仓位
#    - 若持仓期间 CI 突然下降（趋势启动），立即止盈离场
#    - 止盈保守：到达中轴即出，不追求区间另一端
#
#  使用场景：
#    前两套策略找不到趋势信号时，用此策略维持市场活跃度
#    每次只轻仓，盈亏比约 1:1.2（依靠胜率取胜）
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from signal_generator import Signal
from config import RISK_PARAMS

STRATEGY_NAME = "区间高抛低吸"

# ── 策略参数 ──────────────────────────────────
RANGE_PERIOD    = 20     # 区间识别周期（N日高低点）
CI_MIN          = 55.0   # 进入策略的最低震荡指数（必须是震荡市）
CI_TREND_EXIT   = 45.0   # CI降到此值以下→趋势启动，强制离场
EDGE_ATR_MULT   = 1.5    # 进入触发区（距边界 N×ATR 以内）
STOP_ATR_MULT   = 1.0    # 止损距边界外的 ATR 倍数
RSI_OVERSOLD    = 35     # RSI超卖阈值（做多）
RSI_OVERBOUGHT  = 65     # RSI超买阈值（做空）
VOL_CONTRACT    = 0.80   # 量能萎缩阈值（背离确认）
HOLD_MAX_BARS   = 15     # 最长持仓 Bar 数（区间内不能久拖）
# ─────────────────────────────────────────────


def _ci_at(df: pd.DataFrame, i: int, n: int = 14) -> float:
    if i < n:
        return 50.0
    sub = df.iloc[i - n + 1: i + 1]
    tr = pd.concat([
        sub["high"] - sub["low"],
        (sub["high"] - sub["close"].shift(1)).abs(),
        (sub["low"]  - sub["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    sum_tr = tr.sum()
    hl = sub["high"].max() - sub["low"].min()
    if hl <= 0 or sum_tr <= 0:
        return 50.0
    return float(np.clip(100 * np.log10(sum_tr / hl) / np.log10(n), 0, 100))


def generate_signals(symbol: str, df: pd.DataFrame) -> list[Signal]:
    """
    区间高抛低吸策略逐 Bar 扫描。

    入场检查清单（做多，到支撑做多）：
      □ CI > CI_MIN（确认震荡区）
      □ close 在 N日低点上方 EDGE_ATR_MULT×ATR 以内（触及支撑边缘）
      □ RSI < RSI_OVERSOLD（超卖）
      □ volume < VOL_MA5 × VOL_CONTRACT（量能萎缩，卖盘枯竭）
      □ 前一根 K 线不是大阴线（避免追跌）

    入场检查清单（做空，到阻力做空）：对称

    出场规则（任一触发）：
      □ 达到止损
      □ 到达区间中轴（止盈）
      □ 持仓超过 HOLD_MAX_BARS 根 K 线
      □ CI 突然降至 CI_TREND_EXIT（趋势启动信号，赶快离场）
      □ 有浮亏时 MACD 方向背离
    """
    from indicators import add_all_indicators
    df = add_all_indicators(df).dropna(subset=["MA20", "RSI", "ATR"]).reset_index(drop=True)

    signals: list[Signal] = []
    state       = "IDLE"
    direction   = None
    entry_price = stop_loss = target = 0.0
    entry_bar   = 0

    for i in range(RANGE_PERIOD, len(df)):
        r      = df.iloc[i]
        prev   = df.iloc[i - 1]
        date   = r["date"]
        close  = float(r["close"])
        high_  = float(r["high"])
        low_   = float(r["low"])
        atr    = float(r["ATR"])
        rsi    = float(r.get("RSI", 50))
        vol    = float(r.get("volume", 1))
        volma  = float(r.get("VOL_MA5", vol))
        dif    = float(r.get("DIF", 0))
        dea    = float(r.get("DEA", 0))
        p_dif  = float(prev.get("DIF", 0))
        p_dea  = float(prev.get("DEA", 0))

        ci = _ci_at(df, i)

        # ── 区间边界 ──
        n_high = float(df["high"].iloc[i - RANGE_PERIOD: i].max())
        n_low  = float(df["low"].iloc[i - RANGE_PERIOD: i].min())
        midpoint = (n_high + n_low) / 2

        # ── 量能萎缩判断 ──
        vol_shrinking = (volma > 0) and (vol < volma * VOL_CONTRACT)

        # ── 前一根 K 线是否大阴/大阳线（过滤追涨杀跌）──
        prev_body  = abs(float(prev["close"]) - float(prev["open"]))
        prev_range = float(prev["high"]) - float(prev["low"])
        prev_big   = (prev_range > 0) and (prev_body / prev_range > 0.7) and (prev_range > 1.5 * atr)

        # ══════════ 无仓位：寻找区间边缘信号 ══════════
        if state == "IDLE":
            # 必须在震荡市中
            if ci < CI_MIN:
                continue

            # ── 支撑做多 ──
            near_support = (close - n_low) < EDGE_ATR_MULT * atr

            if (near_support
                    and rsi < RSI_OVERSOLD
                    and vol_shrinking
                    and not prev_big):

                stop_loss   = n_low - STOP_ATR_MULT * atr
                target      = midpoint
                entry_price = close
                entry_bar   = i
                direction   = "LONG"
                state       = "IN_TRADE"
                signals.append(Signal(
                    date=date, symbol=symbol, action="BUY",
                    price=close, stop_loss=stop_loss, target=target, atr=atr,
                    reason=(f"区间支撑做多｜支撑={n_low:.1f}｜中轴={midpoint:.1f}"
                            f"｜RSI={rsi:.0f}｜CI={ci:.0f}｜量萎缩{vol/volma:.2f}x")
                ))
                continue

            # ── 阻力做空 ──
            near_resist = (n_high - close) < EDGE_ATR_MULT * atr

            if (near_resist
                    and rsi > RSI_OVERBOUGHT
                    and vol_shrinking
                    and not prev_big):

                stop_loss   = n_high + STOP_ATR_MULT * atr
                target      = midpoint
                entry_price = close
                entry_bar   = i
                direction   = "SHORT"
                state       = "IN_TRADE"
                signals.append(Signal(
                    date=date, symbol=symbol, action="SHORT",
                    price=close, stop_loss=stop_loss, target=target, atr=atr,
                    reason=(f"区间阻力做空｜阻力={n_high:.1f}｜中轴={midpoint:.1f}"
                            f"｜RSI={rsi:.0f}｜CI={ci:.0f}｜量萎缩{vol/volma:.2f}x")
                ))

        # ══════════ 持仓管理 ══════════
        elif state == "IN_TRADE":
            bars_held  = i - entry_bar
            exit_reason = ""

            # ── 通用退出检测 ──
            # 1. CI降低→趋势启动，浮盈或保本即走
            if ci < CI_TREND_EXIT:
                exit_reason = f"趋势启动(CI={ci:.0f}<{CI_TREND_EXIT})，止盈离场"

            # 2. 超时强制止盈/止损
            elif bars_held >= HOLD_MAX_BARS:
                exit_reason = f"持仓超{HOLD_MAX_BARS}根K线，强制平仓"

            # ── 方向特定退出 ──
            if not exit_reason:
                if direction == "LONG":
                    if close <= stop_loss:
                        exit_reason = f"止损({close:.1f}<={stop_loss:.1f})"
                    elif close >= target:
                        exit_reason = f"到达中轴止盈({close:.1f}>={target:.1f})"
                    # MACD 死叉 + 还没到中轴 → 提前认亏/保本
                    elif (dif < dea and p_dif >= p_dea
                            and close < entry_price
                            and bars_held > 3):
                        exit_reason = "MACD死叉且浮亏，提前止损"

                elif direction == "SHORT":
                    if close >= stop_loss:
                        exit_reason = f"止损({close:.1f}>={stop_loss:.1f})"
                    elif close <= target:
                        exit_reason = f"到达中轴止盈({close:.1f}<={target:.1f})"
                    elif (dif > dea and p_dif <= p_dea
                            and close > entry_price
                            and bars_held > 3):
                        exit_reason = "MACD金叉且浮亏，提前止损"

            if exit_reason:
                action = "SELL" if direction == "LONG" else "COVER"
                signals.append(Signal(date, symbol, action, close, 0, 0, atr, exit_reason))
                state = "IDLE"; direction = None

    return signals
