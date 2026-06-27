# ============================================================
#  策略二：突破追涨（横盘突破入场）
#  核心逻辑：
#    ① 识别横盘整理（CI 长期偏高 / N 日价格区间收窄）
#    ② 确认突破（收盘价突破 N 日最高价，实体强劲）
#    ③ 量能配合（成交量 > 1.5×均量）
#    ④ 止损 = 突破位 - 0.5×ATR（突破位即变支撑）
#    ⑤ 止盈 = 突破位 + 整理区间等比投射（最少 2:1）
#
#  过滤条件：
#    - 近 N 日 CI 均值 > 50（说明存在足够的整理积蓄）
#    - 突破 K 线实体率 > 0.5（不能是长影线假突破）
#    - 密度惩罚：突破后若 CI 快速回升则提前止盈
#    - 次日回踩不破突破位 = 二次确认入场机会
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from signal_generator import Signal
from config import RISK_PARAMS

STRATEGY_NAME = "突破追涨"

# ── 策略参数 ──────────────────────────────────
BO_PERIOD      = 20     # 突破参考周期（N 日新高/新低）
CI_LOOKBACK    = 10     # 突破前 CI 观察窗口
CI_BASE        = 50.0   # 突破前 CI 均值须 > 此值（有充分整理）
VOL_EXPAND     = 1.50   # 突破时量能须 > 1.5×均量
BODY_MIN       = 0.50   # 突破 K 线最低实体率（排除长影线假突破）
RETEST_BARS    = 3      # 突破后允许回踩确认的最大 Bar 数
# ─────────────────────────────────────────────


def _ci_series(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """计算全序列 Choppiness Index。"""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    sum_tr = tr.rolling(n).sum()
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    denom = (hh - ll).replace(0, np.nan)
    return (100 * np.log10(sum_tr / denom) / np.log10(n)).clip(0, 100)


def generate_signals(symbol: str, df: pd.DataFrame) -> list[Signal]:
    """
    突破追涨策略逐 Bar 扫描，返回 Signal 列表。

    三种入场时机：
      A. 突破当日直接追涨（强势突破，实体率高 + 量能大）
      B. 突破次日缩量回踩不破突破位（稳妥入场）
      C. 对称：跌破 N 日新低做空
    """
    from indicators import add_all_indicators
    df = add_all_indicators(df).dropna(subset=["MA20", "ATR"]).reset_index(drop=True)

    ci_all = _ci_series(df)
    sl_mult = RISK_PARAMS["atr_stop_mult"]

    signals: list[Signal] = []
    state        = "IDLE"     # IDLE / RETEST_WAIT / IN_TRADE
    direction    = None
    breakout_lvl = 0.0        # 突破位（新高 or 新低）
    consolidation_lo = 0.0   # 整理区间下沿
    retest_bar   = 0
    entry_price  = stop_loss = target = 0.0

    for i in range(BO_PERIOD + CI_LOOKBACK, len(df)):
        r      = df.iloc[i]
        prev   = df.iloc[i - 1]
        date   = r["date"]
        close  = float(r["close"])
        open_  = float(r["open"])
        high   = float(r["high"])
        low    = float(r["low"])
        atr    = float(r["ATR"])
        vol    = float(r.get("volume", 1))
        volma  = float(r.get("VOL_MA5", vol))
        ci_now = float(ci_all.iloc[i]) if pd.notna(ci_all.iloc[i]) else 50.0

        # ── 近期 CI 均值（整理充分度）──
        ci_window = ci_all.iloc[i - CI_LOOKBACK: i]
        ci_mean   = float(ci_window.mean()) if not ci_window.isna().all() else 50.0

        # ── N 日高点/低点（突破参考）──
        n_high = float(df["high"].iloc[i - BO_PERIOD: i].max())
        n_low  = float(df["low"].iloc[i - BO_PERIOD: i].min())

        # ── K 线实体率 ──
        bar_range = high - low
        body      = abs(close - open_)
        body_ratio = body / bar_range if bar_range > 0 else 0

        # ══════════ 无仓位：寻找突破 ══════════
        if state == "IDLE":
            # ── 向上突破 ──
            if (close > n_high
                    and body_ratio >= BODY_MIN
                    and vol > volma * VOL_EXPAND
                    and ci_mean >= CI_BASE):

                breakout_lvl     = n_high
                consolidation_lo = n_low
                bo_range         = breakout_lvl - consolidation_lo

                risk      = close - (breakout_lvl - 0.5 * atr)
                stop_loss = breakout_lvl - 0.5 * atr
                target    = close + max(bo_range, 2 * risk)

                entry_price = close
                direction   = "LONG"

                # 强势突破：当日直接入场
                if body_ratio >= 0.65 and vol > volma * 2.0:
                    state = "IN_TRADE"
                    signals.append(Signal(
                        date=date, symbol=symbol, action="BUY",
                        price=close, stop_loss=stop_loss, target=target, atr=atr,
                        reason=(f"强势向上突破｜{BO_PERIOD}日新高={n_high:.1f}"
                                f"｜实体率={body_ratio:.0%}｜量能={vol/volma:.1f}x"
                                f"｜CI均值={ci_mean:.0f}")
                    ))
                else:
                    # 等回踩确认
                    state      = "RETEST_WAIT"
                    retest_bar = i

            # ── 向下突破（做空）──
            elif (close < n_low
                    and body_ratio >= BODY_MIN
                    and vol > volma * VOL_EXPAND
                    and ci_mean >= CI_BASE):

                breakout_lvl     = n_low
                consolidation_lo = n_high   # 整理上沿
                bo_range         = consolidation_lo - breakout_lvl

                risk      = (breakout_lvl + 0.5 * atr) - close
                stop_loss = breakout_lvl + 0.5 * atr
                target    = close - max(bo_range, 2 * risk)

                entry_price = close
                direction   = "SHORT"

                if body_ratio >= 0.65 and vol > volma * 2.0:
                    state = "IN_TRADE"
                    signals.append(Signal(
                        date=date, symbol=symbol, action="SHORT",
                        price=close, stop_loss=stop_loss, target=target, atr=atr,
                        reason=(f"强势向下突破｜{BO_PERIOD}日新低={n_low:.1f}"
                                f"｜实体率={body_ratio:.0%}｜量能={vol/volma:.1f}"
                                f"｜CI均值={ci_mean:.0f}")
                    ))
                else:
                    state      = "RETEST_WAIT"
                    retest_bar = i

        # ══════════ 等待回踩确认 ══════════
        elif state == "RETEST_WAIT":
            bars_since_bo = i - retest_bar

            if bars_since_bo > RETEST_BARS:
                # 超时没给确认机会，放弃
                state = "IDLE"; direction = None; continue

            if direction == "LONG":
                # 回踩但不破突破位 + 收回突破位上方
                if low <= breakout_lvl * 1.003 and close > breakout_lvl:
                    risk      = close - (breakout_lvl - 0.5 * atr)
                    stop_loss = breakout_lvl - 0.5 * atr
                    bo_range  = breakout_lvl - consolidation_lo
                    target    = close + max(bo_range, 2 * risk)
                    entry_price = close
                    state = "IN_TRADE"
                    signals.append(Signal(
                        date=date, symbol=symbol, action="BUY",
                        price=close, stop_loss=stop_loss, target=target, atr=atr,
                        reason=f"突破回踩确认｜回踩至{low:.1f}未破突破位{breakout_lvl:.1f}"
                    ))
                # 突破失败（跌回突破位以下且收盘在下方）
                elif close < breakout_lvl - atr:
                    state = "IDLE"; direction = None

            elif direction == "SHORT":
                if high >= breakout_lvl * 0.997 and close < breakout_lvl:
                    risk      = (breakout_lvl + 0.5 * atr) - close
                    stop_loss = breakout_lvl + 0.5 * atr
                    bo_range  = consolidation_lo - breakout_lvl
                    target    = close - max(bo_range, 2 * risk)
                    entry_price = close
                    state = "IN_TRADE"
                    signals.append(Signal(
                        date=date, symbol=symbol, action="SHORT",
                        price=close, stop_loss=stop_loss, target=target, atr=atr,
                        reason=f"突破回踩确认｜反弹至{high:.1f}未破突破位{breakout_lvl:.1f}"
                    ))
                elif close > breakout_lvl + atr:
                    state = "IDLE"; direction = None

        # ══════════ 持仓管理 ══════════
        elif state == "IN_TRADE":
            dif   = float(r.get("DIF", 0))
            dea   = float(r.get("DEA", 0))
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
                # CI 重新高于61.8→行情重新变得混乱，止盈离场
                elif ci_now > 65 and close > entry_price:
                    exit_reason = f"CI={ci_now:.0f}行情变混乱，浮盈止盈"
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
                elif ci_now > 65 and close < entry_price:
                    exit_reason = f"CI={ci_now:.0f}行情变混乱，浮盈止盈"
                if exit_reason:
                    signals.append(Signal(date, symbol, "COVER", close, 0, 0, atr, exit_reason))
                    state = "IDLE"; direction = None

    return signals
