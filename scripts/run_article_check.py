#!/usr/bin/env python
"""研报（新浪 2024-08-12，SSRN-4824172 中文版）精确口径对照实验。

研报配置：噪声带 VM=1/L=14 + 昨收修正；入场只在 10:29/11:29/13:59（次分钟开盘成交）；
止损每分钟检查（对向边界=初始；同向带+VWAP=改进1）；改进2=波动目标杠杆
（σ_target=2%, lev_max=4, σ 用指数14日日波动）；成本单边万1、无滑点。
与我们的默认差异：重入规则（band_state: 带外即入，不要求新跨越）、指数层级、样本期。

实验矩阵：标的(指数000300/000905/000932 + ETF 510300/510500/159845)
× 配置(A 初始 / B 改进1 / C 改进2) × 重入规则(new_cross/band_state)
× 时段(全样本 2017-2025 / 研报对齐 2017-2024.07)。

产物：data/results/article_check/article_check_summary.csv + 控制台对照表。
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from intramom.config import load_base_config  # noqa: E402
from intramom.data_loader import load_data  # noqa: E402
from intramom.etf_loader import load_data_etf  # noqa: E402
from intramom.ext_loader import load_data_ext  # noqa: E402
from intramom.index_loader import load_data_index  # noqa: E402
from intramom.metrics import perf_stats  # noqa: E402
from intramom.panel import PanelData  # noqa: E402
from intramom.engine import run_backtest  # noqa: E402

OUT = ROOT / "data" / "results" / "article_check"
ARTICLE_END = pd.Timestamp("2024-07-31")
INIT = 1_000_000.0

# (标签, 层级, 代码, ETF对应指数代码)
INSTRUMENTS = [
    ("沪深300指数", "index", "000300.SH"),
    ("中证500指数", "index", "000905.SH"),
    ("中证1000指数", "index", "000932.SH"),
    ("510300 ETF", "etf", "510300.SH"),
    ("510500 ETF", "etf", "510500.SH"),
    ("159845 ETF", "etf", "159845.SZ"),
]
ETF_INDEX = {"510300.SH": "000300.SH", "510500.SH": "000905.SH", "159845.SZ": "000932.SH"}

CONFIGS = {
    "A 初始(对向带)": {"stop_mode": "opp_band", "sizing": "fixed"},
    "B 改进1(带+VWAP)": {"stop_mode": "band_vwap", "sizing": "fixed"},
    "C 改进2(B+杠杆2%/4)": {"stop_mode": "band_vwap", "sizing": "vol_target"},
}


def make_panel(level: str, code: str, start: str, end: str) -> PanelData:
    if level == "index":
        bundle, _ = load_data_index(code, start, end, ROOT)
    elif code == "510300.SH":
        bundle = load_data(load_base_config(ROOT).with_overrides(), ROOT)
    elif code == "510500.SH":
        bundle, _ = load_data_ext(code, start, end, ROOT)
    else:
        bundle, _ = load_data_etf(code, start, end, ROOT)
    if level == "etf" and code in ETF_INDEX and code != "510300.SH":
        # 注入真实指数日线（vol_source=index 用）
        idx = _load_idx_daily(ETF_INDEX[code])
        bundle.index_daily = idx
    return PanelData.build(bundle)


def _load_idx_daily(idx_code: str) -> pd.DataFrame:
    p = ROOT / "data" / "processed_ext" / f"index_daily_{idx_code.split('.')[0]}.csv"
    if not p.exists():
        sys.path.insert(0, "scripts")
        from run_vol_target_scan import load_index_daily
        return load_index_daily(idx_code)
    return pd.read_csv(p, parse_dates=["trade_date"])


def seg(ret: pd.Series, hi: pd.Timestamp | None = None) -> dict:
    r = ret if hi is None else ret[ret.index <= hi]
    if len(r) < 20:
        return {"Sharpe": np.nan, "IRR": np.nan, "MDD": np.nan, "TotalReturn": np.nan}
    eq = INIT * np.cumprod(1.0 + r.to_numpy())
    return perf_stats(r.to_numpy(), eq, init=INIT)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_base_config(ROOT).with_overrides(
        grid="custom", grid_points=("10:29", "11:29", "13:59"), stop_freq="1min",
        vol_mult=1.0, lookback=14, slippage_per_share=0.0)  # 研报口径：万1 无滑点

    panels = {}
    for lab, level, code in INSTRUMENTS:
        print(f"[article] 构建面板 {lab} ...", flush=True)
        panels[code] = make_panel(level, code, base.start, base.end)

    jobs = []
    for lab, level, code in INSTRUMENTS:
        for cname, cp in CONFIGS.items():
            for em in ("new_cross", "band_state"):
                cfg = base.with_overrides(symbol=code, entry_mode=em,
                                          vol_source="index", **cp)
                jobs.append((lab, level, code, cname, em, cfg))

    def work(job):
        lab, level, code, cname, em, cfg = job
        res = run_backtest(panels[code], cfg)
        d = res.daily.set_index("date")["ret"]
        full = seg(d)
        art = seg(d, ARTICLE_END)
        row = {"标的": lab, "层级": level, "code": code, "配置": cname, "重入": em,
               "exec_rows": len(res.trades)}
        for k in ("TotalReturn", "IRR", "Sharpe", "MDD"):
            row[f"full_{k}"] = full[k]
            row[f"art_{k}"] = art[k]
        return row

    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(work, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "article_check_summary.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    for lab, level, code in INSTRUMENTS:
        print(f"\n===== {lab}（研报对齐段 ~2024-07 / 全样本 ~2025-12）=====")
        sub = df[df.code == code]
        show = sub[["配置", "重入", "art_IRR", "art_Sharpe", "art_MDD",
                    "full_IRR", "full_Sharpe", "full_MDD", "exec_rows"]]
        print(show.to_string(index=False))
    print(f"\n[article] 产物 -> {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
