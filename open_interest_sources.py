from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd


@dataclass(frozen=True)
class OpenInterestStatus:
    symbol: str
    available: bool
    source: str
    latest_date: str
    latest_open_interest: float
    change_pct: float
    note: str


def has_open_interest(df: pd.DataFrame) -> bool:
    return "open_interest" in df.columns and df["open_interest"].notna().sum() > 2


def summarize_open_interest(symbol: str, df: pd.DataFrame) -> OpenInterestStatus:
    if df is None or df.empty:
        return OpenInterestStatus(symbol, False, "none", "", 0.0, 0.0, "无行情数据")
    if not has_open_interest(df):
        simulated = bool(df.get("is_simulated", pd.Series([False])).iloc[-1]) if "is_simulated" in df.columns else False
        note = "仿真数据不提供真实持仓量" if simulated else "当前行情源未返回 open_interest"
        latest_date = str(pd.to_datetime(df["date"].iloc[-1]).date()) if "date" in df.columns else ""
        return OpenInterestStatus(symbol, False, "missing", latest_date, 0.0, 0.0, note)

    valid = df.dropna(subset=["open_interest"]).copy()
    valid["date"] = pd.to_datetime(valid["date"], errors="coerce")
    valid = valid.dropna(subset=["date"]).sort_values("date")
    latest = float(valid["open_interest"].iloc[-1])
    prev = float(valid["open_interest"].iloc[-2]) if len(valid) >= 2 else latest
    chg_pct = (latest - prev) / prev * 100 if prev > 0 else 0.0
    simulated = bool(valid.get("is_simulated", pd.Series([False])).iloc[-1]) if "is_simulated" in valid.columns else False
    source = "akshare:futures_zh_daily_sina.hold" if not simulated else "simulated"
    return OpenInterestStatus(
        symbol=symbol,
        available=True,
        source=source,
        latest_date=str(valid["date"].iloc[-1].date()),
        latest_open_interest=round(latest, 2),
        change_pct=round(chg_pct, 2),
        note="总持仓量可用" if not simulated else "仿真持仓量仅用于流程演示",
    )


def summarize_all_open_interest(all_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for symbol, df in all_data.items():
        status = summarize_open_interest(symbol, df)
        rows.append({
            "symbol": status.symbol,
            "available": status.available,
            "source": status.source,
            "latest_date": status.latest_date,
            "latest_open_interest": status.latest_open_interest,
            "change_pct": status.change_pct,
            "note": status.note,
        })
    return pd.DataFrame(rows)


def _format_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    text = str(value).replace("-", "").strip()
    if len(text) != 8 or not text.isdigit():
        raise ValueError("date 必须是 YYYYMMDD 或 YYYY-MM-DD")
    return text


def fetch_sina_member_position_rank(contract: str, query_date: str | date | datetime) -> dict[str, pd.DataFrame]:
    """
    查询某个具体合约的会员成交量、多单持仓、空单持仓排名。

    contract 必须是交易所真实合约代码，例如 RB2410、I2409、CU2408。
    RB0 / I0 这类主力连续代码通常只适合行情序列，不适合会员持仓排名。
    """
    clean_contract = contract.strip().upper()
    if clean_contract.endswith("0"):
        raise ValueError("会员持仓排名需要真实合约代码，不能使用 RB0 / I0 这类主力连续代码")

    import akshare as ak

    d = _format_date(query_date)
    return {
        "成交量": ak.futures_hold_pos_sina(symbol="成交量", contract=clean_contract, date=d),
        "多单持仓": ak.futures_hold_pos_sina(symbol="多单持仓", contract=clean_contract, date=d),
        "空单持仓": ak.futures_hold_pos_sina(symbol="空单持仓", contract=clean_contract, date=d),
    }


def open_interest_source_plan() -> list[dict[str, str]]:
    return [
        {
            "level": "首选",
            "source": "akshare.futures_zh_daily_sina",
            "data": "日线总持仓量 hold",
            "usage": "映射为 open_interest，直接进入四维行情持续性判断",
        },
        {
            "level": "补充",
            "source": "akshare.futures_hold_pos_sina",
            "data": "具体合约成交量、多单持仓、空单持仓会员排名",
            "usage": "用于判断主力席位集中度和多空分歧，需要真实合约代码",
        },
        {
            "level": "校验",
            "source": "交易所官网每日统计/持仓排名",
            "data": "上期所/大商所/郑商所/广期所公开数据",
            "usage": "实盘前用于校验 akshare 数据延迟、缺失和换月问题",
        },
    ]


if __name__ == "__main__":
    from data_fetcher import get_all_data

    data = get_all_data(use_cache=True)
    print(summarize_all_open_interest(data).to_string(index=False))
