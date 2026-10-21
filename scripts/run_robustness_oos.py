#!/usr/bin/env python
"""旗舰策略稳健性与样本外验证（第一梯队 6 只 ETF）。

① VM×L 稳健性谱：VM ∈ {1.0,1.5,2.0,2.5,3.0} × L ∈ {10,14,20,30}（h3 网格 + A-L-1 不变），
   全样本 + IS(≤2023) + OOS(2024-2025) 三段指标；检验旗舰格(VM2/L20)是否孤立尖点。
② 留出验证：IS 段选最优格 → 看 OOS 表现；IS/OOS Sharpe 相关性（逐标的 + 合并）。
   （策略日内闭合、无隔夜状态，全样本一次回测后按日切片即等价分段回测。）

产物（data/results/robustness_oos/）：
- <code>_spectrum.csv：每组合 full/IS/OOS 指标
- spectrum_heatmaps.png：6 标的 VM×L Sharpe 热图（全样本）
- oos_scatter.png：IS vs OOS Sharpe 逐组合散点（6 面板 + 相关系数）
- holdout_summary.csv：旗舰格 IS/OOS + IS最优格→OOS + 稳健性指标
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from intramom.config import StrategyConfig, load_base_config
from intramom.data_loader import load_data
from intramom.etf_loader import load_data_etf
from intramom.ext_loader import load_data_ext
from intramom.metrics import perf_stats
from intramom.panel import PanelData
from intramom.engine import run_backtest

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

OUT = ROOT / "data" / "results" / "robustness_oos"
OOS_START = pd.Timestamp("2024-01-01")
TIER1 = [("510300.SH", "沪深300"), ("510500.SH", "中证500"), ("515220.SH", "煤炭"),
         ("159852.SZ", "软件"), ("159845.SZ", "中证1000"), ("159326.SZ", "电网设备")]
VMS = [1.0, 1.5, 2.0, 2.5, 3.0]
LBS = [10, 14, 20, 30]
FLAGSHIP = {"grid": "custom", "grid_points": ("10:29", "11:29", "13:59"),
            "stop_mode": "opp_band", "sizing": "fixed", "stop_freq": "1min"}
INIT = 1_000_000.0


def make_bundle(symbol: str, start: str, end: str):
    if symbol == "510300.SH":
        return load_data(StrategyConfig(symbol=symbol, start=start, end=end), ROOT), {}
    if symbol in ("510500.SH", "159915.SZ"):
        return load_data_ext(symbol, start, end, ROOT)
    return load_data_etf(symbol, start, end, ROOT)


def seg_metrics(dates: np.ndarray, ret: np.ndarray, lo, hi) -> dict:
    m = (dates >= lo) & (dates < hi)
    r = ret[m]
    if len(r) < 20:
        return {"N": int(len(r)), "TotalReturn": np.nan, "IRR": np.nan, "Vol": np.nan,
                "Sharpe": np.nan, "MDD": np.nan}
    eq = INIT * np.cumprod(1.0 + r)
    st = perf_stats(r, eq, init=INIT)
    return {k: st[k] for k in ("N", "TotalReturn", "IRR", "Vol", "Sharpe", "MDD")}


def run_one_combo(panel, cfg) -> tuple[dict, pd.DataFrame]:
    res = run_backtest(panel, cfg)
    dl = res.daily[["date", "ret"]].copy()
    dates = dl["date"].to_numpy()
    r = dl["ret"].to_numpy(float)
    row = {"vol_mult": cfg.vol_mult, "lookback": cfg.lookback,
           "exec_rows": len(res.trades), "end_aum": res.end_aum}
    for tag, lo, hi in [("full", pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")),
                        ("IS", pd.Timestamp("1900-01-01"), OOS_START),
                        ("OOS", OOS_START, pd.Timestamp("2100-01-01"))]:
        st = seg_metrics(dates, r, lo, hi)
        for k, v in st.items():
            row[f"{tag}_{k}"] = v
    return row, dl


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_base_config(ROOT).with_overrides(**FLAGSHIP)
    spectra: dict[str, pd.DataFrame] = {}
    summary_rows = []

    for sym, cname in TIER1:
        code = sym.split(".")[0]
        print(f"[robo] {sym}({cname}) 加载 ...", flush=True)
        bundle, _ = make_bundle(sym, base.start, base.end)
        panel = PanelData.build(bundle)
        combos = [base.with_overrides(symbol=sym, vol_mult=vm, lookback=lb)
                  for vm in VMS for lb in LBS]
        with ThreadPoolExecutor(max_workers=8) as ex:
            outs = list(ex.map(lambda c: run_one_combo(panel, c), combos))
        df = pd.DataFrame([o[0] for o in outs])
        df.insert(0, "code", sym)
        df.insert(1, "name", cname)
        spectra[sym] = df
        df.to_csv(OUT / f"{code}_spectrum.csv", index=False)

        pos_share = (df["full_Sharpe"] > 0).mean()
        fl = df[(df.vol_mult == 2.0) & (df.lookback == 20)].iloc[0]
        rank = int((df["full_Sharpe"] > fl["full_Sharpe"]).sum()) + 1
        # IS/OOS 相关（20 组合）
        valid = df.dropna(subset=["IS_Sharpe", "OOS_Sharpe"])
        corr = valid["IS_Sharpe"].corr(valid["OOS_Sharpe"]) if len(valid) > 3 else np.nan
        if len(valid) > 3:
            is_best = df.loc[df["IS_Sharpe"].idxmax()]
            is_best_lab = f"VM{is_best.vol_mult:g}/L{is_best.lookback}"
            is_best_s, is_best_os, is_best_otr = (is_best["IS_Sharpe"], is_best["OOS_Sharpe"],
                                                  is_best["OOS_TotalReturn"])
        else:  # 无 IS 段（如 159326 上市太晚）
            is_best_lab, is_best_s, is_best_os, is_best_otr = "n/a", np.nan, np.nan, np.nan
        summary_rows.append({
            "code": sym, "name": cname,
            "flagship_full_Sharpe": fl["full_Sharpe"], "flagship_full_TR": fl["full_TotalReturn"],
            "flagship_full_MDD": fl["full_MDD"],
            "flagship_IS_Sharpe": fl["IS_Sharpe"], "flagship_OOS_Sharpe": fl["OOS_Sharpe"],
            "flagship_OOS_TR": fl["OOS_TotalReturn"], "flagship_OOS_MDD": fl["OOS_MDD"],
            "flagship_rank_in_lattice": f"{rank}/{len(df)}",
            "pos_share_full": pos_share,
            "IS_best": is_best_lab,
            "IS_best_Sharpe": is_best_s,
            "IS_best_OOS_Sharpe": is_best_os,
            "IS_best_OOS_TR": is_best_otr,
            "IS_OOS_corr": corr,
        })
        print(f"[robo] {sym} 完成: 旗舰 full={fl['full_Sharpe']:+.3f} IS={fl['IS_Sharpe']:+.3f} "
              f"OOS={fl['OOS_Sharpe']:+.3f} | 谱内为正 {pos_share:.0%} | IS/OOS corr={corr:.2f}",
              flush=True)

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(OUT / "holdout_summary.csv", index=False)

    # ---- 热图（全样本 Sharpe，VM×L） ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (sym, cname) in zip(axes.ravel(), TIER1):
        df = spectra[sym]
        piv = df.pivot(index="lookback", columns="vol_mult", values="full_Sharpe")
        im = ax.imshow(piv.to_numpy(), cmap="RdYlGn", aspect="auto",
                       vmin=-1.0, vmax=max(1.2, np.nanmax(piv.to_numpy())))
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{v:g}" for v in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel("VM")
        ax.set_ylabel("L")
        ax.set_title(f"{cname} {sym.split('.')[0]}")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.to_numpy()[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="black")
        # 旗舰格标框
        j = list(piv.columns).index(2.0)
        i = list(piv.index).index(20)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="blue", lw=2))
    fig.suptitle("VM×L Sharpe 稳健性谱（h3 A-L-1，全样本；蓝框=旗舰格 VM2/L20）")
    fig.tight_layout()
    fig.savefig(OUT / "spectrum_heatmaps.png")
    plt.close(fig)

    # ---- IS vs OOS 散点 ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    all_is, all_oos = [], []
    for ax, (sym, cname) in zip(axes.ravel(), TIER1):
        df_all = spectra[sym]
        df = df_all.dropna(subset=["IS_Sharpe", "OOS_Sharpe"])
        ax.scatter(df["IS_Sharpe"], df["OOS_Sharpe"], s=22, alpha=0.75, color="steelblue")
        lim = [-1.5, 2.0]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        fl = df_all[(df_all.vol_mult == 2.0) & (df_all.lookback == 20)].iloc[0]
        if not np.isnan(fl["IS_Sharpe"]):
            ax.scatter([fl["IS_Sharpe"]], [fl["OOS_Sharpe"]], s=70, marker="*",
                       color="red", zorder=5, label="旗舰格")
            ax.legend(fontsize=7)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("IS Sharpe (≤2023)")
        ax.set_ylabel("OOS Sharpe (2024-25)")
        ax.set_title(f"{cname} {sym.split('.')[0]}")
        ax.grid(alpha=0.3)
        all_is.extend(df["IS_Sharpe"].tolist())
        all_oos.extend(df["OOS_Sharpe"].tolist())
    pooled_corr = pd.Series(all_is).corr(pd.Series(all_oos))
    fig.suptitle(f"IS vs OOS Sharpe（20 组合/标的；合并 Pearson r={pooled_corr:.2f}）")
    fig.tight_layout()
    fig.savefig(OUT / "oos_scatter.png")
    plt.close(fig)

    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print("\n===== 留出验证汇总 =====")
    print(summ.to_string(index=False))
    print(f"\n[robo] 产物 -> {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
