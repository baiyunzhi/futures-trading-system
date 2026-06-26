# ============================================================
#  信号质量验证脚本
#  用法：python verify_signals.py
#        python verify_signals.py --symbols RB0 CU0 M0
#        python verify_signals.py --force   (强制重新拉取，忽略缓存)
#
#  输出：
#    ① akshare 连通性检测
#    ② 各品种数据摘要（真实/仿真、行数、日期范围）
#    ③ 三套策略信号质量报告
#    ④ 具体交易记录（入场日期、价格、止损、结果）
#    ⑤ 策略横向对比汇总表
# ============================================================

from __future__ import annotations
import sys
import argparse
import logging
import textwrap
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

# ─────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────

SEP  = "-" * 72
SEP2 = "=" * 72

def header(title: str):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)

def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def err(msg):  print(f"  [ERR]  {msg}")


# ─────────────────────────────────────────────
#  Step 1: akshare 连通性检测
# ─────────────────────────────────────────────

def check_akshare() -> bool:
    section("Step 1 / 5   akshare 连通性检测")
    try:
        import akshare as ak
        ok(f"akshare 已安装，版本 = {ak.__version__}")
    except ImportError:
        warn("akshare 未安装，将使用缓存或仿真数据继续验证")
        return False

    # 用一个轻量接口测试网络
    try:
        import akshare as ak
        test = ak.futures_zh_daily_sina(symbol="RB0")
        if test is not None and not test.empty:
            ok(f"网络连通，RB0 返回 {len(test)} 行数据")
            ok(f"列名示例：{list(test.columns[:6])}")
            return True
        else:
            warn("RB0 返回空数据，可能是交易所接口限流，稍后重试")
            return False
    except Exception as e:
        err(f"网络请求失败：{e}")
        warn("将使用仿真数据继续验证策略逻辑")
        return False


# ─────────────────────────────────────────────
#  Step 2: 数据获取与摘要
# ─────────────────────────────────────────────

def fetch_data(symbols: list[str], force: bool) -> dict[str, pd.DataFrame]:
    section("Step 2 / 5   行情数据获取")
    from data_fetcher import get_data
    from indicators import add_all_indicators
    from config import ALL_SYMBOLS

    result = {}
    for sym in symbols:
        name = ALL_SYMBOLS.get(sym, sym)
        try:
            df = get_data(sym, use_cache=not force)
            df = add_all_indicators(df)
            is_sim = bool(df.get("is_simulated", pd.Series([True])).iloc[0])
            tag = "[仿真]" if is_sim else "[真实]"
            date_rng = f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}"
            ok(f"{name:6s}({sym})  {tag}  {len(df):4d}行  {date_rng}")
            result[sym] = df
        except Exception as e:
            err(f"{name}({sym})  获取失败：{e}")
    return result


# ─────────────────────────────────────────────
#  Step 3: 单品种信号扫描
# ─────────────────────────────────────────────

def _run_strategy(name: str, fn, symbol: str, df: pd.DataFrame) -> dict:
    """对单品种运行一套策略，返回信号统计。"""
    try:
        signals = fn(symbol, df)
    except Exception as e:
        return {"error": str(e)}

    entries = [s for s in signals if s.action in ("BUY", "SHORT")]
    exits   = [s for s in signals if s.action in ("SELL", "COVER")]

    if not entries:
        return {
            "total": 0, "entries": 0, "exits": 0,
            "longs": 0, "shorts": 0, "trades": []
        }

    # 配对计算盈亏
    # 修复：按时间顺序遍历用状态机一开一平配对。
    # 原 zip(entries, exits) 在出现悬空开仓（最后未平仓）时会整体错位，导致盈亏算错。
    trades = []
    open_sig = None   # 当前持有的开仓信号；None 表示空仓
    for s in signals:
        if s.action in ("BUY", "SHORT"):
            if open_sig is None:
                open_sig = s
        elif s.action in ("SELL", "COVER") and open_sig is not None:
            ent, ext = open_sig, s
            if ent.action == "BUY":
                pnl_pct = (ext.price - ent.price) / ent.price * 100
                direction = "多"
            else:
                pnl_pct = (ent.price - ext.price) / ent.price * 100
                direction = "空"
            days_held = (ext.date - ent.date).days if hasattr(ext.date - ent.date, "days") else 0
            trades.append({
                "entry_date":  ent.date.date() if hasattr(ent.date, "date") else ent.date,
                "exit_date":   ext.date.date() if hasattr(ext.date, "date") else ext.date,
                "direction":   direction,
                "entry_price": ent.price,
                "stop_loss":   ent.stop_loss,
                "target":      ent.target,
                "exit_price":  ext.price,
                "pnl_pct":     round(pnl_pct, 2),
                "days_held":   days_held,
                "win":         pnl_pct > 0,
                "entry_reason": ent.reason,
                "exit_reason":  ext.reason,
            })
            open_sig = None

    win_trades  = [t for t in trades if t["win"]]
    loss_trades = [t for t in trades if not t["win"]]
    win_rate    = len(win_trades) / len(trades) * 100 if trades else 0
    avg_win     = np.mean([t["pnl_pct"] for t in win_trades])  if win_trades  else 0
    avg_loss    = np.mean([t["pnl_pct"] for t in loss_trades]) if loss_trades else 0
    profit_factor = (abs(avg_win * len(win_trades)) /
                     max(abs(avg_loss * len(loss_trades)), 1e-9))

    return {
        "total":    len(signals),
        "entries":  len(entries),
        "exits":    len(exits),
        "longs":    sum(1 for s in entries if s.action == "BUY"),
        "shorts":   sum(1 for s in entries if s.action == "SHORT"),
        "trades":   trades,
        "win_rate": round(win_rate, 1),
        "avg_win":  round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_days": round(np.mean([t["days_held"] for t in trades]), 1) if trades else 0,
    }


def report_strategy(strategy_name: str, fn, all_data: dict, show_trades: bool = True):
    """打印单策略完整报告。"""
    section(f"策略：{strategy_name}")
    from config import ALL_SYMBOLS

    summary_rows = []
    all_trades   = []

    for sym, df in all_data.items():
        name = ALL_SYMBOLS.get(sym, sym)
        res  = _run_strategy(strategy_name, fn, sym, df)
        if "error" in res:
            err(f"{name}({sym}) 运行失败：{res['error']}")
            continue

        if res["entries"] == 0:
            warn(f"{name:6s}({sym})  无信号")
            continue

        ok(f"{name:6s}({sym})  "
           f"信号={res['entries']:2d}笔  "
           f"多={res['longs']}空={res['shorts']}  "
           f"胜率={res['win_rate']:5.1f}%  "
           f"均盈={res['avg_win']:+.2f}%  "
           f"均亏={res['avg_loss']:+.2f}%  "
           f"盈亏比={res['profit_factor']:.2f}  "
           f"均持{res['avg_days']:.0f}天")

        for t in res["trades"]:
            t["symbol"] = sym
            t["name"]   = name
            all_trades.append(t)

        summary_rows.append({
            "品种":   f"{name}({sym})",
            "笔数":   res["entries"],
            "胜率%":  res["win_rate"],
            "均盈%":  res["avg_win"],
            "均亏%":  res["avg_loss"],
            "盈亏比": res["profit_factor"],
            "均持天": res["avg_days"],
        })

    if not all_trades:
        warn("本策略在所有品种上均无完整配对信号")
        return pd.DataFrame(summary_rows)

    # 打印具体交易明细
    if show_trades:
        print(f"\n  {'品种':<8} {'方向':<4} {'入场日':<12} {'出场日':<12} "
              f"{'入场价':>8} {'出场价':>8} {'盈亏%':>7} {'持天':>5}  出场原因")
        print(f"  {'-'*8} {'-'*4} {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*7} {'-'*5}  {'-'*20}")
        for t in sorted(all_trades, key=lambda x: str(x["entry_date"])):
            pnl_str  = f"{t['pnl_pct']:+.2f}%"
            win_mark = "WIN" if t["win"] else "LOSS"
            reason   = t["exit_reason"][:28] if t["exit_reason"] else "-"
            print(f"  {t['name']:6s}({t['symbol']:4s}) "
                  f"{t['direction']:<4} {str(t['entry_date']):<12} {str(t['exit_date']):<12} "
                  f"{t['entry_price']:>8.1f} {t['exit_price']:>8.1f} "
                  f"{pnl_str:>7} {t['days_held']:>5}  {win_mark} {reason}")

        wins  = [t for t in all_trades if t["win"]]
        total_pnl = sum(t["pnl_pct"] for t in all_trades)
        print(f"\n  汇总：{len(all_trades)} 笔  "
              f"胜率 {len(wins)/len(all_trades)*100:.1f}%  "
              f"合计盈亏 {total_pnl:+.2f}%  "
              f"平均 {total_pnl/len(all_trades):+.2f}%/笔")

    return pd.DataFrame(summary_rows)


# ─────────────────────────────────────────────
#  Step 4: 三套策略汇总对比
# ─────────────────────────────────────────────

def compare_strategies(all_data: dict):
    section("Step 4 / 5   三套策略横向对比汇总")
    import strategy_trend    as st
    import strategy_breakout as sb
    import strategy_range    as sr

    strategies = [
        ("趋势跟踪",    st.generate_signals),
        ("突破追涨",    sb.generate_signals),
        ("区间高抛低吸", sr.generate_signals),
    ]

    compare_rows = []
    for name, fn in strategies:
        all_trades = []
        for sym, df in all_data.items():
            res = _run_strategy(name, fn, sym, df)
            if "error" in res or not res["trades"]:
                continue
            all_trades.extend(res["trades"])

        if not all_trades:
            compare_rows.append({
                "策略": name, "总笔数": 0, "胜率%": 0,
                "均盈%": 0, "均亏%": 0, "盈亏比": 0, "均持天": 0, "合计%": 0
            })
            continue

        wins = [t for t in all_trades if t["win"]]
        loss = [t for t in all_trades if not t["win"]]
        compare_rows.append({
            "策略":   name,
            "总笔数": len(all_trades),
            "胜率%":  round(len(wins)/len(all_trades)*100, 1),
            "均盈%":  round(np.mean([t["pnl_pct"] for t in wins]),  2) if wins else 0,
            "均亏%":  round(np.mean([t["pnl_pct"] for t in loss]),  2) if loss else 0,
            "盈亏比": round((abs(np.mean([t["pnl_pct"] for t in wins]))  * len(wins)) /
                            max(abs(np.mean([t["pnl_pct"] for t in loss])) * len(loss), 1e-9), 2) if wins and loss else 0,
            "均持天": round(np.mean([t["days_held"] for t in all_trades]), 1),
            "合计%":  round(sum(t["pnl_pct"] for t in all_trades), 2),
        })

    cdf = pd.DataFrame(compare_rows)
    if cdf.empty:
        warn("无有效对比数据")
        return

    # 格式化打印
    cols = ["策略", "总笔数", "胜率%", "均盈%", "均亏%", "盈亏比", "均持天", "合计%"]
    col_w = [12, 8, 8, 8, 8, 8, 8, 8]
    header_str = "".join(f"{c:>{w}}" for c, w in zip(cols, col_w))
    print(f"\n  {header_str}")
    print(f"  {'-'*sum(col_w)}")
    for _, row in cdf.iterrows():
        vals = [
            f"{row['策略']}", f"{row['总笔数']:.0f}",
            f"{row['胜率%']:.1f}%", f"{row['均盈%']:+.2f}%",
            f"{row['均亏%']:+.2f}%", f"{row['盈亏比']:.2f}",
            f"{row['均持天']:.0f}天", f"{row['合计%']:+.2f}%",
        ]
        row_str = "".join(f"{v:>{w}}" for v, w in zip(vals, col_w))
        print(f"  {row_str}")

    # 策略互补性分析
    print()
    for _, row in cdf.iterrows():
        if row["总笔数"] == 0:
            warn(f"{row['策略']}：无信号（可能当前市场不符合触发条件）")
        elif row["胜率%"] >= 50 and row["盈亏比"] >= 1.5:
            ok(f"{row['策略']}：胜率+盈亏比均达标，信号质量良好")
        elif row["胜率%"] >= 55:
            ok(f"{row['策略']}：胜率较高，适合当前行情")
        elif row["盈亏比"] >= 2.0:
            ok(f"{row['策略']}：盈亏比优秀，少量高质量信号")
        else:
            warn(f"{row['策略']}：当前数据集下表现待验证，建议扩大样本")


# ─────────────────────────────────────────────
#  Step 5: 当前实盘建议（最新信号）
# ─────────────────────────────────────────────

def show_live_signals(all_data: dict):
    section("Step 5 / 5   当前实盘信号（基于最新行情）")
    import strategy_trend    as st
    import strategy_breakout as sb
    import strategy_range    as sr
    from config import ALL_SYMBOLS

    strategies = [
        ("趋势跟踪",    st.generate_signals),
        ("突破追涨",    sb.generate_signals),
        ("区间高抛低吸", sr.generate_signals),
    ]

    any_signal = False
    for sym, df in all_data.items():
        name = ALL_SYMBOLS.get(sym, sym)
        sym_signals = []
        for strat_name, fn in strategies:
            sigs = fn(sym, df)
            entries = [s for s in sigs if s.action in ("BUY", "SHORT")]
            if entries:
                last = entries[-1]
                # 只显示最近 30 天内的信号
                days_ago = (pd.Timestamp.now() - pd.Timestamp(last.date)).days
                if days_ago <= 30:
                    sym_signals.append((strat_name, last))

        if sym_signals:
            any_signal = True
            for strat_name, sig in sym_signals:
                direction = "做多" if sig.action == "BUY" else "做空"
                print(f"  [{strat_name}] {name}({sym}) {direction}")
                print(f"    日期={sig.date.date()}  入场={sig.price:.1f}  "
                      f"止损={sig.stop_loss:.1f}  目标={sig.target:.1f}")
                print(f"    {sig.reason}")
                print()

    if not any_signal:
        warn("当前30天内无新鲜实盘信号")
        print("  说明：信号触发条件严格是正常的，请继续观察或接入真实行情")


# ─────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="交易系统信号质量验证")
    parser.add_argument("--symbols", nargs="+",
                        default=["RB0", "CU0", "M0", "Y0", "I0"],
                        help="要验证的品种代码，默认取5个代表性品种")
    parser.add_argument("--force",   action="store_true",
                        help="忽略缓存，强制重新拉取行情")
    parser.add_argument("--all",     action="store_true",
                        help="验证config中全部15个品种（较慢）")
    parser.add_argument("--no-trades", action="store_true",
                        help="不打印具体交易明细，只显示汇总")
    args = parser.parse_args()

    header(f"交易系统信号质量验证  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Step 1: 连通性
    ak_ok = check_akshare()
    if not ak_ok:
        print("\n  → 将使用仿真数据验证策略逻辑，信号数量会偏少，属正常现象")

    # Step 2: 数据
    from config import ALL_SYMBOLS
    if args.all:
        symbols = list(ALL_SYMBOLS.keys())
    else:
        symbols = [s.upper() for s in args.symbols]

    all_data = fetch_data(symbols, force=args.force)
    if not all_data:
        err("无有效数据，退出")
        sys.exit(1)

    show_trades = not args.no_trades

    # Step 3: 逐策略报告
    section("Step 3 / 5   各策略信号详情")
    import strategy_trend    as st
    import strategy_breakout as sb
    import strategy_range    as sr

    for name, fn in [
        ("趋势跟踪",    st.generate_signals),
        ("突破追涨",    sb.generate_signals),
        ("区间高抛低吸", sr.generate_signals),
    ]:
        report_strategy(name, fn, all_data, show_trades=show_trades)

    # Step 4: 汇总对比
    compare_strategies(all_data)

    # Step 5: 实盘信号
    show_live_signals(all_data)

    header("验证完成")
    print("  如需查看可视化仪表板：python main.py → http://127.0.0.1:8050")
    print()


if __name__ == "__main__":
    main()
