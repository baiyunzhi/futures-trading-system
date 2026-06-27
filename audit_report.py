from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AuditItem:
    level: str
    status: str
    title: str
    detail: str


def build_audit_items(simulated: bool | None = None) -> list[AuditItem]:
    data_status = "待确认" if simulated is None else ("风险" if simulated else "通过")
    data_detail = "当前数据包含仿真行情，不能作为实盘信号。" if simulated else "当前缓存标记为真实行情；实盘前仍需校验换月、夜盘和复权。"
    return [
        AuditItem("高", "已修复", "真实成交价重算风控", "次日开盘成交后按实际成交价重算止损、止盈和 ATR 风险距离。"),
        AuditItem("高", "已修复", "OHLCV 数据质量校验", "过滤 open/high/low/close/volume 缺失、非正数、high/low 关系错误和重复日期。"),
        AuditItem("高", "部分修复", "组合资金口径", "单品种回测已修复；组合展示仍是多品种对比，不等同真实统一账户组合。"),
        AuditItem("高", data_status, "真实行情来源", data_detail),
        AuditItem("中", "已修复", "模拟盘日内熔断", "日内亏损触发后停止接收新信号，并按当日收盘强制平掉已有持仓。"),
        AuditItem("中", "已修复", "仿真数据可复现", "仿真种子改为稳定哈希，跨进程生成结果一致。"),
        AuditItem("中", "待完善", "日内成交顺序", "日线同时触及止损和止盈时仍按保守止损优先；需分钟级数据才能还原真实先后。"),
        AuditItem("低", "待完善", "交易所细则", "保证金、涨跌停、交易日历、品种手续费仍是简化模型。"),
    ]


def audit_items_to_frame(items: list[AuditItem]) -> pd.DataFrame:
    return pd.DataFrame([item.__dict__ for item in items])


def has_simulated_data(all_data: dict[str, pd.DataFrame]) -> bool:
    for df in all_data.values():
        if "is_simulated" in df.columns and df["is_simulated"].astype(bool).any():
            return True
    return False
