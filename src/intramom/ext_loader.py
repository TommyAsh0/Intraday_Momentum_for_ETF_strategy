"""扩展标的（510500.SH / 159915.SZ）数据加载：前序项目 raw 分钟 CSV + Tushare 日线 → DataBundle。

与 data_loader（510300 冻结快照）的差异：
- 分钟：前序 raw/ext_etf_min_<sym>_<year>.csv（列：ts_code,trade_time,OHLC,vol,amount），
  首用构建后缓存 parquet 到 data/processed_ext/（幂等，二次秒读）
- 日线：Tushare fund_daily（pre_close 含除息调整口径），缓存到 data/processed_ext/
- 可交易性：合成全真（ETF 涨停封板极端罕见；510300 全样本仅 8 分钟。QA 中登记该近似）
- 不完整日处理：缺 09:30 竞价 bar（240 根）保留（day_open 回退首根有效 bar）；
  缺任何 09:31..15:00 价格标签的日剔除并登记（σ 面板不允许 NaN）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import EXPECTED_LABELS, DataBundle, label_key

EXT_RAW = Path(r"D:\quantitative_project\复现工程\Intraday_Momentum_for_ETF\data\raw")
EXT_PROC_DIR = "data/processed_ext"

EXT_SYMBOLS = {
    "510500.SH": {"raw_prefix": "ext_etf_min_510500_SH", "name": "中证500ETF南方"},
    "159915.SZ": {"raw_prefix": "ext_etf_min_159915_SZ", "name": "易方达创业板ETF"},
}


def _load_ext_minute(symbol: str, project_root: str | Path) -> pd.DataFrame:
    """读取/构建扩展标的分钟数据（带 parquet 缓存）。"""
    cache_dir = Path(project_root) / EXT_PROC_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"minute_bars_{symbol.split('.')[0]}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    for year in range(2017, 2026):
        f = EXT_RAW / f"{EXT_SYMBOLS[symbol]['raw_prefix']}_{year}.csv"
        if f.exists():
            frames.append(pd.read_csv(f, parse_dates=["trade_time"]))
    if not frames:
        raise FileNotFoundError(f"未找到 {symbol} 的 raw 分钟数据: {EXT_RAW}")
    mn = pd.concat(frames, ignore_index=True).sort_values("trade_time").reset_index(drop=True)
    mn["trade_date"] = mn["trade_time"].dt.normalize()
    mn["hhmm"] = mn["trade_time"].dt.strftime("%H:%M")
    mn = mn[["ts_code", "trade_time", "trade_date", "hhmm",
             "open", "high", "low", "close", "vol", "amount"]]
    mn.to_parquet(cache, index=False)
    return mn


def _load_ext_daily(symbol: str, project_root: str | Path) -> pd.DataFrame:
    """读取/构建扩展标的日线（Tushare fund_daily，带缓存）。"""
    cache_dir = Path(project_root) / EXT_PROC_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"daily_{symbol.split('.')[0]}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["trade_date"])
    from .options import init_tushare
    pro = init_tushare()
    df = pro.fund_daily(ts_code=symbol, start_date="20150101", end_date="20251231",
                        fields="ts_code,trade_date,open,high,low,close,pre_close,"
                               "change,pct_chg,vol,amount")
    if df.empty:
        raise RuntimeError(f"fund_daily 未取到 {symbol}")
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


def load_data_ext(symbol: str, start: str, end: str,
                  project_root: str | Path) -> tuple[DataBundle, dict]:
    """扩展标的 → DataBundle（结构对齐 PanelData.build 要求）+ QA。"""
    if symbol not in EXT_SYMBOLS:
        raise ValueError(f"未登记的扩展标的 {symbol}，可选 {list(EXT_SYMBOLS)}")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    mn = _load_ext_minute(symbol, project_root)
    mn = mn[(mn.trade_date >= start_ts) & (mn.trade_date <= end_ts)].copy()
    mn = mn.sort_values("trade_time").reset_index(drop=True)
    daily = _load_ext_daily(symbol, project_root)

    # --- 结构校验与不完整日处理 ---
    labels = sorted(mn.hhmm.unique(), key=label_key)
    if labels != EXPECTED_LABELS:
        raise AssertionError(f"{symbol} bar 标签集合异常: {len(labels)} 个")
    price_labels = set(EXPECTED_LABELS[1:])
    bars = mn.groupby("trade_date")
    keep_days, drop_days = [], []
    for d, gdf in bars:
        labs = set(gdf.hhmm)
        if price_labels - labs:
            drop_days.append((d, len(gdf)))
        else:
            keep_days.append(d)  # 缺 09:30（240 根）保留
    if drop_days:
        mn = mn[mn.trade_date.isin(keep_days)].reset_index(drop=True)

    # 分钟聚合与日频交叉锚定
    days = pd.DatetimeIndex(sorted(mn.trade_date.unique()))
    c15 = mn[mn.hhmm == "15:00"].set_index("trade_date")["close"].reindex(days)
    dly = daily.set_index("trade_date")
    d_close = (c15 - dly["close"].reindex(days)).abs().max()

    # 合成可交易性（全真；近似登记）
    trad = mn[["trade_date", "hhmm"]].copy()
    trad["可买"] = 1
    trad["可卖"] = 1
    trad["是否有成交"] = np.where(mn["vol"].fillna(0) > 0, 1, 0)
    trad["交易时间"] = trad["hhmm"].str[:2] + trad["hhmm"].str[3:]
    trad["label"] = trad["hhmm"]

    qa = {"symbol": symbol, "n_days": len(days),
          "first_day": str(days[0].date()), "last_day": str(days[-1].date()),
          "dropped_days_missing_price_labels": [(str(d.date()), n) for d, n in drop_days],
          "close_maxdiff_vs_daily": float(d_close),
          "tradability": "synthetic all-true (ETF 封板近似忽略); 零成交 bar 双边封锁",
          "zero_vol_bars": int((trad["是否有成交"] == 0).sum())}

    bundle = DataBundle(
        minute=mn, daily=daily, index_daily=daily.copy(), tradability=trad,
        data_dir=Path(project_root) / EXT_PROC_DIR,
        source_files={"minute": f"minute_bars_{symbol.split('.')[0]}.parquet",
                      "daily": f"daily_{symbol.split('.')[0]}.csv",
                      "tradability": "synthetic"})
    return bundle, qa
