"""数据加载：读取前序项目冻结的 510300 数据快照并做结构校验。

数据源（只读引用，不复制）：
- minute_bars_510300.parquet：1 分钟 bar（09:30 竞价 bar + END 标签 09:31..11:30 / 13:01..15:00）
- daily_510300.csv：日线（pre_close 为除息调整口径；ret = pct_chg/100）
- index_daily_000300.csv：对照基准（仅报告用）
- tradability.csv：分钟级可买/可卖/是否有成交
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def label_key(s: str) -> int:
    return int(s[:2]) * 60 + int(s[3:])


# 标准 bar 标签集合（END 标签，241 个/日）
EXPECTED_LABELS = (
    ["09:30"] + [f"09:{m:02d}" for m in range(31, 60)]
    + [f"10:{m:02d}" for m in range(60)] + [f"11:{m:02d}" for m in range(31)]
    + [f"13:{m:02d}" for m in range(1, 60)] + [f"14:{m:02d}" for m in range(60)]
    + ["15:00"]
)

# 决策时点 13:00 的信号 bar 用上午收盘 bar 11:30（无 13:00 标签）
SIG_BAR = {"13:00": "11:30"}


@dataclass
class DataBundle:
    """原始数据包（尚未面板化）。"""
    minute: pd.DataFrame      # trade_time/trade_date/hhmm/OHLC/vol/amount
    daily: pd.DataFrame       # 全历史日线（含样本前预热段）
    index_daily: pd.DataFrame
    tradability: pd.DataFrame
    data_dir: Path
    source_files: dict[str, str]


def resolve_data_dir(cfg_data_dir: str, project_root: str | Path) -> Path:
    p = Path(cfg_data_dir)
    if not p.is_absolute():
        p = Path(project_root) / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"数据目录不存在: {p}（可在 config 中修改 data_dir）")
    return p


def load_data(cfg, project_root: str | Path) -> DataBundle:
    """加载冻结数据并按配置样本区间裁剪分钟数据，做结构与锚点校验。"""
    data_dir = resolve_data_dir(cfg.data_dir, project_root)
    start, end = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)

    mn = pd.read_parquet(data_dir / "minute_bars_510300.parquet")
    daily = pd.read_csv(data_dir / "daily_510300.csv", parse_dates=["trade_date"])
    idx_daily = pd.read_csv(data_dir / "index_daily_000300.csv", parse_dates=["trade_date"])
    trad = pd.read_csv(data_dir / "tradability.csv", dtype={"交易日期": str, "交易时间": str})

    mn = mn[(mn.trade_date >= start) & (mn.trade_date <= end)].copy()
    mn = mn.sort_values("trade_time").reset_index(drop=True)

    # --- 结构校验：标签集合与每日 bar 数 ---
    labels = sorted(mn.hhmm.unique(), key=label_key)
    if labels != EXPECTED_LABELS:
        raise AssertionError(f"bar 标签集合异常: {len(labels)} 个")
    bars_per_day = mn.groupby("trade_date").size()
    incomplete = bars_per_day[bars_per_day != 241]
    known_gap = {"20170809": 240}  # 唯一已知异常日：缺 09:30 竞价 bar（DT-001 已登记）
    got = {d.strftime("%Y%m%d"): int(v) for d, v in incomplete.items()}
    if got != known_gap:
        raise AssertionError(f"不完整日异常: {got}")

    # --- 分钟聚合与日频交叉锚定（close/open 应一致）---
    days = pd.DatetimeIndex(sorted(mn.trade_date.unique()))
    c15 = mn[mn.hhmm == "15:00"].set_index("trade_date")["close"].reindex(days)
    dly = daily.set_index("trade_date")
    d_close = (c15 - dly["close"].reindex(days)).abs().max()
    if not d_close < 0.002:
        raise AssertionError(f"分钟聚合 close 与日频偏差过大: {d_close}")

    # --- tradability 规范化 ---
    trad = trad.copy()
    trad["label"] = trad["交易时间"].str[:2] + ":" + trad["交易时间"].str[2:]
    trad["trade_date"] = pd.to_datetime(trad["交易日期"], format="%Y%m%d")

    return DataBundle(
        minute=mn, daily=daily, index_daily=idx_daily, tradability=trad,
        data_dir=data_dir,
        source_files={
            "minute": "minute_bars_510300.parquet",
            "daily": "daily_510300.csv",
            "index_daily": "index_daily_000300.csv",
            "tradability": "tradability.csv",
        },
    )
