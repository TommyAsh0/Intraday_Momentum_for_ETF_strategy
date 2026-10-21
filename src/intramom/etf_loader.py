"""行业/大盘 ETF 通用数据加载：LAN MySQL（etf_min + fund_daily）→ DataBundle（带本地缓存）。

与 ext_loader 的关系：ext_loader 面向"前序项目 raw CSV 扩展标的"；本模块面向任意 ETF 代码，
直接从 MySQL 抓取并缓存到 data/etf_scan_data/<code>/（幂等，二次秒读）。
510300.SH 走前序冻结快照（data_loader），510500.SH/159915.SZ 走 ext_loader——本模块用于其余标的。

口径与结构校验（与 510300 冻结数据一致）：
- 分钟：09:30 竞价 bar + END 标签 09:31..11:30 / 13:01..15:00（241 根/日）
- 日线：fund_daily（pre_close 除息调整口径；ret = pct_chg/100）
- 缺 09:30 竞价 bar（240 根）保留；缺任何 09:31..15:00 价格标签的日剔除并登记
- 可交易性：合成全真 + 零成交 bar 双边封锁（ETF 封板近似忽略，QA 登记）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import EXPECTED_LABELS, DataBundle, label_key

MYSQL_LIB = Path(r"D:\quantitative_project\复现工程\-mysql_data")
SCAN_DATA_DIR = "data/etf_scan_data"


def _pro_api():
    if str(MYSQL_LIB) not in sys.path:
        sys.path.insert(0, str(MYSQL_LIB))
    import data_driven as dd
    return dd.pro_api()


def load_etf_minute(symbol: str, project_root: str | Path,
                    start_year: int = 2017, end_year: int = 2025) -> pd.DataFrame:
    """分钟数据（按年分块抓取，parquet 缓存）。"""
    cache_dir = Path(project_root) / SCAN_DATA_DIR / symbol.split(".")[0]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "minute_bars.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    pro = _pro_api()
    frames = []
    try:
        for y in range(start_year, end_year + 1):
            df = pro.etf_min(ts_code=symbol, start_date=f"{y}0101",
                             end_date=f"{y}1231", limit=100000)
            if len(df):
                frames.append(df)
                print(f"[etf_loader] {symbol} {y}: {len(df)} rows", flush=True)
    finally:
        pro.close()
    if not frames:
        raise RuntimeError(f"etf_min 无数据: {symbol}")
    mn = pd.concat(frames, ignore_index=True).sort_values("trade_time").reset_index(drop=True)
    mn["trade_date"] = mn["trade_time"].dt.normalize()
    mn["hhmm"] = mn["trade_time"].dt.strftime("%H:%M")
    keep = ["ts_code", "trade_time", "trade_date", "hhmm",
            "open", "high", "low", "close", "vol", "amount"]
    mn = mn[[c for c in keep if c in mn.columns]]
    mn.to_parquet(cache, index=False)
    return mn


def load_etf_daily(symbol: str, project_root: str | Path) -> pd.DataFrame:
    """日线（fund_daily，csv 缓存）。"""
    cache_dir = Path(project_root) / SCAN_DATA_DIR / symbol.split(".")[0]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "daily.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["trade_date"])
    pro = _pro_api()
    try:
        df = pro.fund_daily(ts_code=symbol, start_date="20150101", end_date="20251231",
                            limit=100000)
    finally:
        pro.close()
    if df.empty:
        raise RuntimeError(f"fund_daily 无数据: {symbol}")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


def load_data_etf(symbol: str, start: str, end: str,
                  project_root: str | Path) -> tuple[DataBundle, dict]:
    """任意 ETF → DataBundle（结构对齐 PanelData.build 要求）+ QA。"""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    mn = load_etf_minute(symbol, project_root)
    mn = mn[(mn.trade_date >= start_ts) & (mn.trade_date <= end_ts)].copy()
    mn = mn.sort_values("trade_time").reset_index(drop=True)
    daily = load_etf_daily(symbol, project_root)

    labels = sorted(mn.hhmm.unique(), key=label_key)
    if labels != EXPECTED_LABELS:
        raise AssertionError(f"{symbol} bar 标签集合异常: {len(labels)} 个")
    price_labels = set(EXPECTED_LABELS[1:])
    keep_days, drop_days = [], []
    for d, gdf in mn.groupby("trade_date"):
        labs = set(gdf.hhmm)
        if price_labels - labs:
            drop_days.append((d, len(gdf)))
        else:
            keep_days.append(d)
    if drop_days:
        mn = mn[mn.trade_date.isin(keep_days)].reset_index(drop=True)
    if mn.empty:
        raise RuntimeError(f"{symbol} 在 {start}~{end} 无完整分钟数据（{len(drop_days)} 日均不完整）")

    days = pd.DatetimeIndex(sorted(mn.trade_date.unique()))
    # --- 日线缺口由分钟聚合填充（MySQL fund_daily 全局源缺口：2021-06-22、2021-12-17、
    #     2023-04-27、2023-10-24、2024-12-13、2025-06-24，与前序 510300 冻结口径一致：
    #     填充日 pre_close = 前一交易日收盘）---
    dly_idx = pd.DatetimeIndex(daily["trade_date"])
    gap_days = [d for d in days if d not in dly_idx]
    if gap_days:
        fills = []
        for d in gap_days:
            g = mn[mn.trade_date == d]
            first_o = g["open"].dropna().iloc[0]
            fills.append({"ts_code": symbol, "trade_date": d, "open": first_o,
                          "high": g["high"].max(), "low": g["low"].min(),
                          "close": g.loc[g.hhmm == "15:00", "close"].iloc[0],
                          "vol": g["vol"].sum(), "amount": g["amount"].sum() / 1000.0,
                          "source": "minute_filled"})
        daily = pd.concat([daily, pd.DataFrame(fills)], ignore_index=True)
        daily = daily.sort_values("trade_date").reset_index(drop=True)
        # 填充 pre_close / change / pct_chg（pre_close = 前一交易日收盘）
        daily["pre_close"] = daily["close"].shift(1).where(daily["pre_close"].isna(),
                                                         daily["pre_close"])
        daily["change"] = daily["close"] - daily["pre_close"]
        daily["pct_chg"] = (daily["close"] / daily["pre_close"] - 1.0) * 100.0

    c15 = mn[mn.hhmm == "15:00"].set_index("trade_date")["close"].reindex(days)
    dly = daily.set_index("trade_date")
    d_close = (c15 - dly["close"].reindex(days)).abs().max()

    trad = mn[["trade_date", "hhmm"]].copy()
    trad["可买"] = 1
    trad["可卖"] = 1
    trad["是否有成交"] = np.where(mn["vol"].fillna(0) > 0, 1, 0)
    trad["交易时间"] = trad["hhmm"].str[:2] + trad["hhmm"].str[3:]
    trad["label"] = trad["hhmm"]

    qa = {"symbol": symbol, "n_days": len(days),
          "first_day": str(days[0].date()), "last_day": str(days[-1].date()),
          "dropped_days_missing_price_labels": [(str(d.date()), n) for d, n in drop_days],
          "daily_gap_filled_from_minute": [str(d.date()) for d in gap_days],
          "close_maxdiff_vs_daily": float(d_close),
          "tradability": "synthetic all-true + 零成交 bar 双边封锁",
          "zero_vol_bars": int((trad["是否有成交"] == 0).sum()),
          "zero_vol_share": float((trad["是否有成交"] == 0).mean())}

    bundle = DataBundle(
        minute=mn, daily=daily, index_daily=daily.copy(), tradability=trad,
        data_dir=Path(project_root) / SCAN_DATA_DIR / symbol.split(".")[0],
        source_files={"minute": "minute_bars.parquet", "daily": "daily.csv",
                      "tradability": "synthetic"})
    return bundle, qa
