"""指数层级数据加载：LAN MySQL index_min → DataBundle（研报口径复现用）。

与 ETF 通道的差异：
- 指数分钟无成交量（vol=None），只有 amount——按 vol ≈ amount/close/100 合成估计手数，
  使 VWAP 机制可直接复用（分钟均价 ≈ 该分钟真实 VWAP，文档登记该近似）
- 可交易性全部合成全真（指数仅作信号级研究，不可直接交易）
- 日线：index_daily（pre_close 口径；已知源缺口 2020-01-10/2020-02-27 由前序 000300 处理过，
  其余指数由调用方用交易日历 ffill 处理，见 run_vol_target_scan.load_index_daily 同款思路）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import EXPECTED_LABELS, DataBundle, label_key

MYSQL_LIB = Path(r"D:\quantitative_project\复现工程\-mysql_data")
INDEX_DATA_DIR = "data/index_scan_data"


def _pro_api():
    if str(MYSQL_LIB) not in sys.path:
        sys.path.insert(0, str(MYSQL_LIB))
    import data_driven as dd
    return dd.pro_api()


def load_index_minute(idx_code: str, project_root: str | Path,
                      start_year: int = 2017, end_year: int = 2025) -> pd.DataFrame:
    cache_dir = Path(project_root) / INDEX_DATA_DIR / idx_code.split(".")[0]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "minute_bars.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    pro = _pro_api()
    frames = []
    try:
        for y in range(start_year, end_year + 1):
            df = pro.index_min(ts_code=idx_code, start_date=f"{y}0101",
                               end_date=f"{y}1231", limit=100000)
            if len(df):
                frames.append(df)
                print(f"[index_loader] {idx_code} {y}: {len(df)} rows", flush=True)
    finally:
        pro.close()
    if not frames:
        raise RuntimeError(f"index_min 无数据: {idx_code}")
    mn = pd.concat(frames, ignore_index=True).sort_values("trade_time").reset_index(drop=True)
    mn["trade_date"] = mn["trade_time"].dt.normalize()
    mn["hhmm"] = mn["trade_time"].dt.strftime("%H:%M")
    mn = mn[["ts_code", "trade_time", "trade_date", "hhmm",
             "open", "high", "low", "close", "vol", "amount"]]
    mn.to_parquet(cache, index=False)
    return mn


def load_index_daily_df(idx_code: str, project_root: str | Path) -> pd.DataFrame:
    """指数日线（index_daily + 开市日历对齐 ffill，缺口日 pct_chg=0）。"""
    cache_dir = Path(project_root) / INDEX_DATA_DIR / idx_code.split(".")[0]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "daily.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["trade_date"])
    pro = _pro_api()
    try:
        df = pro.index_daily(ts_code=idx_code, start_date="20100101", end_date="20251231",
                             limit=100000)
    finally:
        pro.close()
    if df.empty:
        raise RuntimeError(f"index_daily 无数据: {idx_code}")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    cal = pd.read_csv(Path(project_root).parent / "Intraday_Momentum_for_ETF" / "data"
                      / "processed" / "trade_calendar.csv", dtype={"cal_date": str})
    open_days = pd.to_datetime(cal.loc[cal.is_open == 1, "cal_date"]).tolist()
    df = (df.set_index("trade_date").reindex(open_days).ffill().reset_index()
          .rename(columns={"index": "trade_date", "cal_date": "trade_date"}))
    df["pct_chg"] = df["close"].pct_change().fillna(0.0) * 100.0
    df.to_csv(cache, index=False)
    return df


def load_data_index(idx_code: str, start: str, end: str,
                    project_root: str | Path) -> tuple[DataBundle, dict]:
    """指数 → DataBundle（vol 为 amount/close/100 合成估计，可交易性全真）。"""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    mn = load_index_minute(idx_code, project_root)
    mn = mn[(mn.trade_date >= start_ts) & (mn.trade_date <= end_ts)].copy()
    mn = mn.sort_values("trade_time").reset_index(drop=True)
    daily = load_index_daily_df(idx_code, project_root)

    labels = sorted(mn.hhmm.unique(), key=label_key)
    if labels != EXPECTED_LABELS:
        raise AssertionError(f"{idx_code} bar 标签集合异常: {len(labels)} 个")
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

    # 合成成交量（分钟均价近似）：vol ≈ amount / close / 100（手）
    mn["vol"] = mn["amount"] / mn["close"] / 100.0

    days = pd.DatetimeIndex(sorted(mn.trade_date.unique()))
    c15 = mn[mn.hhmm == "15:00"].set_index("trade_date")["close"].reindex(days)
    dly = daily.set_index("trade_date")
    d_close = (c15 - dly["close"].reindex(days)).abs().max()

    trad = mn[["trade_date", "hhmm"]].copy()
    trad["可买"] = 1
    trad["可卖"] = 1
    trad["是否有成交"] = 1
    trad["交易时间"] = trad["hhmm"].str[:2] + trad["hhmm"].str[3:]
    trad["label"] = trad["hhmm"]

    qa = {"symbol": idx_code, "n_days": len(days),
          "first_day": str(days[0].date()), "last_day": str(days[-1].date()),
          "dropped_days_missing_price_labels": [(str(d.date()), n) for d, n in drop_days],
          "close_maxdiff_vs_daily": float(d_close),
          "tradability": "synthetic all-true（指数信号级研究）",
          "vol": "synthetic vol ≈ amount/close/100（指数无成交量）"}
    bundle = DataBundle(
        minute=mn, daily=daily, index_daily=daily.copy(), tradability=trad,
        data_dir=Path(project_root) / INDEX_DATA_DIR / idx_code.split(".")[0],
        source_files={"minute": "minute_bars.parquet", "daily": "daily.csv",
                      "tradability": "synthetic", "vol": "synthetic from amount"})
    return bundle, qa
