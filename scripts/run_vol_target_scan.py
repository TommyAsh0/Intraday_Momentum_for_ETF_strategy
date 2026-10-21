#!/usr/bin/env python
"""σ_target 精细遍历（波动目标仓位专项研究），多标的版本。

固定旗舰口径（h3: 10:29/11:29/13:59 入场 + L-1 分钟止损，VM2/L20），
遍历维度：σ_target × max_leverage × vol_source(etf/index) × 变体(C/D)。

标的与对标指数：
- 510300.SH → 000300.SH（前序冻结）
- 510500.SH → 000905.SH（MySQL index_daily，注入面板）
- 159845.SZ → 000932.SH（同上）

用法：
    python scripts/run_vol_target_scan.py                          # 510300，C 和 D，双 σ 源
    python scripts/run_vol_target_scan.py --symbols 510500.SH 159845.SZ
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from intramom import load_base_config, run_sweep  # noqa: E402
from intramom.config import StrategyConfig  # noqa: E402
from intramom.data_loader import load_data  # noqa: E402
from intramom.ext_loader import EXT_SYMBOLS, load_data_ext  # noqa: E402
from intramom.etf_loader import load_data_etf  # noqa: E402
from intramom.panel import PanelData  # noqa: E402
from intramom.runner import VARIANT_PARAMS  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

SIGMA_TARGETS = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.04]
MAX_LEVS = [2.0, 4.0]
FIXED = {"grid": "custom", "grid_points": ("10:29", "11:29", "13:59"),
         "stop_freq": "1min", "vol_mult": 2.0, "lookback": 20}
INDEX_MAP = {"510300.SH": "000300.SH", "510500.SH": "000905.SH", "159845.SZ": "000932.SH"}
OUT = ROOT / "data" / "results"


def load_index_daily(idx_code: str) -> pd.DataFrame:
    """对标指数日线（MySQL index_daily，缓存；源缺口日 2020-01-10/2020-02-27 前推填充，
    缺口日 ret=0——与前序 510300 用 index_min 填充等价的保守处理）。"""
    cache = ROOT / "data" / "processed_ext" / f"index_daily_{idx_code.split('.')[0]}.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["trade_date"])
    else:
        sys.path.insert(0, r"D:\quantitative_project\复现工程\-mysql_data")
        import data_driven as dd
        pro = dd.pro_api()
        try:
            df = pro.index_daily(ts_code=idx_code, start_date="20150101",
                                 end_date="20251231", limit=100000)
        finally:
            pro.close()
        if df.empty:
            raise RuntimeError(f"index_daily 无数据: {idx_code}")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").reset_index(drop=True)
    # 对齐开市日历并前推填充收盘价，重算 pct_chg（缺口日 ret=0）
    cal = pd.read_csv(ROOT.parent / "Intraday_Momentum_for_ETF" / "data" / "processed"
                      / "trade_calendar.csv", dtype={"cal_date": str})
    open_days = pd.to_datetime(cal.loc[cal.is_open == 1, "cal_date"]).tolist()
    df = df.set_index("trade_date").reindex(open_days).ffill().reset_index()
    df = df.rename(columns={"index": "trade_date", "cal_date": "trade_date"})
    df["pct_chg"] = df["close"].pct_change().fillna(0.0) * 100.0
    df.to_csv(cache, index=False)
    return df


def make_panel(symbol: str, start: str, end: str) -> PanelData:
    """构建面板；对 ext 标的注入真实对标指数日线（替换占位）。"""
    if symbol == "510300.SH":
        bundle = load_data(StrategyConfig(symbol=symbol, start=start, end=end), ROOT)
    elif symbol in EXT_SYMBOLS:
        bundle, _ = load_data_ext(symbol, start, end, ROOT)
    else:
        bundle, _ = load_data_etf(symbol, start, end, ROOT)
    idx_code = INDEX_MAP[symbol]
    idx_df = load_index_daily(idx_code)
    bundle.index_daily = idx_df  # 真实指数日线（含 pct_chg）
    return PanelData.build(bundle)


def main() -> int:
    ap = argparse.ArgumentParser(description="σ_target × lev × vol_source 遍历（多标的）")
    ap.add_argument("--symbols", nargs="+", default=["510300.SH"],
                    choices=list(INDEX_MAP))
    ap.add_argument("--variants", nargs="+", default=["C", "D"], choices=["C", "D"])
    args = ap.parse_args()

    base = load_base_config(ROOT).with_overrides(**FIXED)
    all_rows = []
    for sym in args.symbols:
        code = sym.split(".")[0]
        print(f"[sigmat] {sym} 构建面板 ...", flush=True)
        panel = make_panel(sym, base.start, base.end)
        # σ 源 sanity：确认 index 源与 etf 源确实不同
        s_etf = panel.sigma_daily(14, "etf")
        s_idx = panel.sigma_daily(14, "index")
        import numpy as np
        print(f"          σ 源差异: corr={np.corrcoef(s_etf, s_idx)[0,1]:.5f} "
              f"mean|Δrel|={np.mean(np.abs(s_idx - s_etf) / s_etf):.2%}", flush=True)
        for v in args.variants:
            for src in ("etf", "index"):
                cfg = base.with_overrides(symbol=sym, vol_source=src, **VARIANT_PARAMS[v])
                name = f"sigmat_scan_{code}_{v}-L-1_{src}"
                df = run_sweep(cfg, {"sigma_target": SIGMA_TARGETS,
                                     "max_leverage": MAX_LEVS},
                               project_root=ROOT, sweep_name=name, n_workers=8,
                               panel=panel)
                df["variant"] = v
                df["vol_source"] = src
                df["symbol"] = sym
                all_rows.append(df)
                print(f"[sigmat] {code} {v}-L-1 σ源={src}: {len(df)} 组合完成", flush=True)
    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(OUT / "sweeps" / f"sigmat_scan_{'_'.join(s.split('.')[0] for s in args.symbols)}_all.csv",
                index=False)

    # ---- 图：每个标的一行（C/D），σ 源分线 ----
    n_sym = len(args.symbols)
    n_var = len(args.variants)
    fig, axes = plt.subplots(n_sym, n_var, figsize=(6.5 * n_var, 3.6 * n_sym),
                             squeeze=False)
    for r, sym in enumerate(args.symbols):
        code = sym.split(".")[0]
        for c, v in enumerate(args.variants):
            ax = axes[r][c]
            sub = full[(full.symbol == sym) & (full.variant == v)]
            for src, mk in (("etf", "o-"), ("index", "s--")):
                for lev in MAX_LEVS:
                    s = sub[(sub.vol_source == src) & (sub.max_leverage == lev)]
                    s = s.sort_values("sigma_target")
                    ax.plot(s["sigma_target"] * 100, s["Sharpe"], mk, ms=4,
                            label=f"{src} lev≤{lev:g}")
            ax.axvline(2.0, color="gray", ls=":", lw=1.0)
            ax.set_title(f"{code} {v}-L-1", fontsize=10)
            ax.set_xlabel("σ_target (%)")
            ax.set_ylabel("Sharpe")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
    fig.suptitle("σ_target 精细遍历（h3 + L-1，VM2/L20）")
    fig.tight_layout()
    fig.savefig(OUT / "sweeps" / f"sigmat_scan_{'_'.join(s.split('.')[0] for s in args.symbols)}.png")
    plt.close(fig)

    # ---- 汇总 ----
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    for sym in args.symbols:
        for v in args.variants:
            for src in ("etf", "index"):
                sub = full[(full.symbol == sym) & (full.variant == v)
                           & (full.vol_source == src)]
                best = sub.loc[sub.Sharpe.idxmax()]
                base_row = sub[(sub.sigma_target == 0.02) & (sub.max_leverage == 4.0)]
                b_txt = (f"论文点(2%,4): Sharpe={base_row.Sharpe.iloc[0]:+.3f}"
                         if len(base_row) else "")
                print(f"{sym} {v}-L-1 σ源={src}: 最优 σt={best.sigma_target:g}/"
                      f"lev{best.max_leverage:g} Sharpe={best.Sharpe:+.3f} "
                      f"TR={best.TotalReturn:+.3f} MDD={best.MDD:.3f} | {b_txt}")
    print(f"\n[sigmat] 产物 -> {OUT / 'sweeps'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
