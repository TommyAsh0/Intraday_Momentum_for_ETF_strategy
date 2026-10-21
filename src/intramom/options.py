"""期权数据层：多标的 ETF 期权合约表与日行情的抓取、QA 与本地冻结。

标的（UNDERLYINGS）：
- 300etf：SSE 300ETF 期权（标的 510300.SH，2019-12-23 上市）
- 500etf：SSE 500ETF 期权（标的 510500.SH，2022-09-19 上市）
- cyb：SZSE 创业板ETF 期权（标的 159915.SZ，2022-09-19 上市）

数据源：Tushare Pro（token 从 数据库构建与更新/config.env 读取，不落任何凭据到产物）。
产物（data/options/，按标的分文件）：
- opt_basic_<key>.csv：合约表（含 per_unit，除息调整合约单位非标准）
- opt_daily_<key>.csv：日行情（trade_date/ts_code/close/settle/vol/oi）
- qa_option_data.json：QA 报告（每日存活合约数 vs 返回行数等）

抓取策略（已探测验证）：合约表按 call_put 分取规避 12000 行上限；
日行情按交易日逐日取（已实证：当日存活合约数 == 返回行数，无截断），断点续抓。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

TUSHARE_CONFIG = Path(r"D:\quantitative_project\Tushare\数据库构建与更新\config.env")
OPTIONS_DIR = "data/options"

UNDERLYINGS: dict[str, dict] = {
    "300etf": {"exchange": "SSE", "keyword": "300ETF", "ts_code": "510300.SH",
               "opt_start": "20191223"},
    "500etf": {"exchange": "SSE", "keyword": "500ETF", "ts_code": "510500.SH",
               "opt_start": "20220919"},
    "cyb": {"exchange": "SZSE", "keyword": "创业板ETF", "ts_code": "159915.SZ",
            "opt_start": "20220919"},
}

DAILY_COLS = ["ts_code", "trade_date", "close", "settle", "vol", "oi"]


def basic_file(key: str) -> str:
    return f"opt_basic_{key}.csv"


def daily_file(key: str) -> str:
    return f"opt_daily_{key}.csv"


def init_tushare(config_path: str | Path = TUSHARE_CONFIG):
    """初始化 Tushare Pro API（token 只进内存，不打印不落盘）。"""
    import dotenv
    import tushare as ts
    dotenv.load_dotenv(str(config_path))
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError(f"未在 {config_path} 找到 TUSHARE_TOKEN")
    ts.set_token(token)
    return ts.pro_api()


def fetch_basic(pro, key: str) -> pd.DataFrame:
    """指定标的期权合约表：C/P 分取以规避行数上限，按 name 关键词过滤。"""
    spec = UNDERLYINGS[key]
    frames = []
    for cp in ("C", "P"):
        b = pro.opt_basic(exchange=spec["exchange"], call_put=cp,
                          fields="ts_code,name,call_put,exercise_price,maturity_date,"
                                 "list_date,delist_date,per_unit")
        frames.append(b)
        time.sleep(0.3)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["name"].str.contains(spec["keyword"], na=False)].copy()
    df = df.sort_values(["maturity_date", "call_put", "exercise_price"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"未取到 {key}({spec['keyword']}) 期权合约")
    return df


def _fetch_daily_one_day(pro, exchange: str, yyyymmdd: str) -> pd.DataFrame:
    return pro.opt_daily(trade_date=yyyymmdd, exchange=exchange,
                         fields="ts_code,trade_date,close,settle,vol,oi")


def fetch_daily(pro, key: str, trade_days: list[str], codes: set[str],
                out_csv: Path, sleep_s: float = 0.25,
                max_retry: int = 6, log_every: int = 100) -> dict:
    """逐日抓取日行情并过滤到本标的合约集合，增量追加写出（断点续抓）。"""
    done = set()
    if out_csv.exists():
        old = pd.read_csv(out_csv, dtype={"trade_date": str})
        done = set(old["trade_date"].unique())
        del old
    todo = [d for d in trade_days if d not in done]
    stats = {"underlying": key, "days_total": len(trade_days),
             "days_done_before": len(done), "days_fetched": 0, "rows_written": 0,
             "retries": 0, "failed_days": []}
    header = not out_csv.exists()
    t0 = time.time()
    for i, d in enumerate(todo):
        df = None
        for attempt in range(max_retry):
            try:
                df = _fetch_daily_one_day(pro, UNDERLYINGS[key]["exchange"], d)
                break
            except Exception:
                stats["retries"] += 1
                if attempt == max_retry - 1:
                    stats["failed_days"].append(d)
                else:
                    time.sleep(min(2 ** attempt, 20))
        if df is None:
            continue
        sub = df[df.ts_code.isin(codes)]
        if len(sub):
            sub.to_csv(out_csv, mode="a", header=header, index=False)
            header = False
            stats["rows_written"] += len(sub)
        stats["days_fetched"] += 1
        if log_every and (i + 1) % log_every == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            print(f"[opt_daily {key}] {i + 1}/{len(todo)} days, rows+={stats['rows_written']}, "
                  f"{rate:.1f} d/s, eta={(len(todo) - i - 1) / max(rate, 1e-9):.0f}s", flush=True)
        time.sleep(sleep_s)
    return stats


def qa_option_data(basic: pd.DataFrame, daily: pd.DataFrame,
                   trade_days: list[str]) -> dict:
    """QA：每日返回行数应等于当日存活合约数（上市~退市区间内）。"""
    alive = (
        basic.assign(key=1)
        .merge(pd.DataFrame({"trade_date": trade_days, "key": 1}), on="key")
        .query("list_date <= trade_date <= delist_date")
        .groupby("trade_date").size().rename("alive")
    )
    got = daily.groupby("trade_date").size().rename("rows")
    cmp_ = pd.concat([alive, got], axis=1).fillna(0).astype(int)
    bad = cmp_[cmp_.alive != cmp_.rows]
    return {
        "n_contracts": int(len(basic)),
        "maturity_range": [str(basic.maturity_date.min()), str(basic.maturity_date.max())],
        "per_unit_values": {str(k): int(v) for k, v in basic.per_unit.value_counts().items()},
        "n_days_daily": int(daily.trade_date.nunique()),
        "n_rows_daily": int(len(daily)),
        "days_mismatch_alive_vs_rows": int(len(bad)),
        "mismatch_examples": bad.head(10).reset_index().to_dict("records"),
        "settle_le0_rows": int((daily.settle <= 0).sum()),
        "oi_na_rows": int(daily.oi.isna().sum()),
    }


def build_option_data(project_root: str | Path, key: str, trade_days: list[str],
                      sleep_s: float = 0.25, basic_only: bool = False) -> dict:
    """端到端（单标的）：合约表 + 日行情抓取 + QA + 冻结。返回 QA 报告。"""
    root = Path(project_root)
    out_dir = root / OPTIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pro = init_tushare()

    basic_path = out_dir / basic_file(key)
    if basic_path.exists():
        basic = pd.read_csv(basic_path, dtype=str)
    else:
        basic = fetch_basic(pro, key)
        basic.to_csv(basic_path, index=False)
    qa = {"underlying": key, "basic_rows": len(basic)}
    if basic_only:
        return qa

    daily_path = out_dir / daily_file(key)
    stats = fetch_daily(pro, key, trade_days, set(basic.ts_code), daily_path, sleep_s=sleep_s)
    daily = pd.read_csv(daily_path, dtype={"trade_date": str})
    qa.update(qa_option_data(basic, daily, trade_days))
    qa["fetch_stats"] = stats
    # QA 汇总文件（多标的合并）
    qa_path = out_dir / "qa_option_data.json"
    all_qa = {}
    if qa_path.exists():
        with open(qa_path, encoding="utf-8") as f:
            all_qa = json.load(f)
    all_qa[key] = qa
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2, default=str)
    return qa


def load_option_data(project_root: str | Path, key: str = "300etf") -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取本地冻结的期权数据（不联网）。"""
    out_dir = Path(project_root) / OPTIONS_DIR
    basic = pd.read_csv(out_dir / basic_file(key), dtype=str)
    daily = pd.read_csv(out_dir / daily_file(key), dtype={"trade_date": str})
    for c in ("close", "settle", "vol", "oi"):
        daily[c] = pd.to_numeric(daily[c], errors="coerce")
    for c in ("exercise_price", "per_unit"):
        basic[c] = pd.to_numeric(basic[c], errors="coerce")
    return basic, daily
