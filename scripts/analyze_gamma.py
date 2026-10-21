#!/usr/bin/env python
"""GAMMA 状态（GEX）对日内动量策略的影响分析。

链路：期权日行情 → BS 反解 IV → 日度 GEX（做市商惯例，call + / put −，元/1%）
→ 与策略日收益（默认 A_g15_VM1_L14）按 t−1 对齐（无前视）→ 区间对比 + 回归。

分析内容：
1) 正/负 GEX 区间：日收益均值/t 检验/Sharpe/胜率/合计
2) 回归：ret ~ GEX；ret ~ 1[GEX<0]；分段斜率 ret ~ β+·GEX⁺ + β−·GEX⁻
3) GEX 三档（低/中/高）收益对比
4) 逐笔口径：开仓日对应区间下的收盘收益/MAE 对比
5) 图：GEX 时序+策略净值 / 散点+分段拟合 / 三档柱图 / 逐笔对比

产物（写入 run 目录，前缀 gex_）：gex_regime.csv / gex_regressions.csv /
gex_terciles.csv / gex_per_trade_regime.csv / gex_analysis.png；
GEX 日度面板缓存于 data/options/gex_daily.csv。
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
import numpy as np
import pandas as pd

from intramom.gamma import build_gex_daily
from intramom.metrics import ols_multi, ols_stats
from intramom.options import load_option_data

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

GEX_CACHE = ROOT / "data" / "options" / "gex_daily.csv"
DEFAULT_RUN = ROOT / "data" / "results" / "sweeps" / "vm_L_g15_g05" / "A_g15_VM1_L14_c1bp_s1t"


def load_or_build_gex() -> tuple[pd.DataFrame, dict]:
    """读缓存的日度 GEX；不存在则由期权冻结数据现算并缓存。"""
    if GEX_CACHE.exists():
        return pd.read_csv(GEX_CACHE, parse_dates=["date"]), {}
    basic, daily = load_option_data(ROOT)
    daily_etf = pd.read_csv(ROOT.parent / "Intraday_Momentum_for_ETF" / "data" / "processed"
                            / "daily_510300.csv", parse_dates=["trade_date"])
    exdiv = pd.read_csv(ROOT.parent / "Intraday_Momentum_for_ETF" / "data" / "processed"
                        / "exdiv_days_510300.csv")
    gex, qa = build_gex_daily(basic, daily, daily_etf, exdiv)
    gex.to_csv(GEX_CACHE, index=False)
    return gex, qa


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """两样本 Welch t 检验（均值差）。返回 (t, p)。"""
    import math
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # 用正态近似（df 通常很大）
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return t, p


def main() -> int:
    ap = argparse.ArgumentParser(description="GAMMA 状态对策略收益的影响分析")
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    print("[gex] 加载/构建日度 GEX 面板 ...")
    gex, qa = load_or_build_gex()
    if qa:
        print("[gex] 面板 QA:", {k: qa[k] for k in
                               ("n_days", "date_range", "gex_pos_days", "gex_neg_days",
                                "iv_fail_share", "gex_yi_range")})

    eq = pd.read_csv(run_dir / "equity_daily.csv", parse_dates=["date"])
    df = eq[["date", "ret"]].merge(gex[["date", "gex_yi", "gex", "iv_median", "S"]],
                                   on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    df["gex_yi_lag"] = df["gex_yi"].shift(1)          # t−1 收盘 GEX，无前视
    df = df.dropna(subset=["gex_yi_lag"]).reset_index(drop=True)
    df["ret_bps"] = df["ret"] * 1e4
    df["neg"] = (df["gex_yi_lag"] < 0).astype(int)
    print(f"[gex] 对齐样本: {len(df)} 日 ({df.date.min().date()}..{df.date.max().date()}) "
          f"负GEX日占 {df.neg.mean():.1%}")

    # ---- 1) 正/负区间对比 ----
    pos = df.loc[df.neg == 0, "ret_bps"].to_numpy()
    neg = df.loc[df.neg == 1, "ret_bps"].to_numpy()
    t_diff, p_diff = welch_t(neg, pos)
    regime = pd.DataFrame([
        {"regime": "GEX>0(long gamma)", "n": len(pos), "mean_bps": pos.mean(),
         "std_bps": pos.std(ddof=1), "sharpe": pos.mean() / pos.std(ddof=1) * np.sqrt(252),
         "hit": (pos > 0).mean(), "sum_bps": pos.sum()},
        {"regime": "GEX<0(short gamma)", "n": len(neg), "mean_bps": neg.mean(),
         "std_bps": neg.std(ddof=1), "sharpe": neg.mean() / neg.std(ddof=1) * np.sqrt(252),
         "hit": (neg > 0).mean(), "sum_bps": neg.sum()},
    ])
    regime.to_csv(run_dir / "gex_regime.csv", index=False)

    # ---- 2) 回归 ----
    y = df["ret_bps"].to_numpy()
    x = df["gex_yi_lag"].to_numpy()
    r1 = ols_stats(y, x)                                   # ret ~ GEX
    r2 = ols_stats(y, df["neg"].to_numpy(float))           # ret ~ 1[GEX<0]
    X = np.column_stack([np.ones(len(df)), np.clip(x, 0, None), np.clip(x, None, 0)])
    r3 = ols_multi(y, X, ["const", "beta_pos(GEX>0)", "beta_neg(GEX<0)"])  # 分段斜率
    regs = pd.DataFrame([
        {"model": "ret ~ GEX", "term": "GEX", "coef": r1["beta"], "t": r1["t_beta"],
         "p": r1["p_beta"], "r2": r1["r2"], "n": r1["n"]},
        {"model": "ret ~ 1[GEX<0]", "term": "1[GEX<0]", "coef": r2["beta"], "t": r2["t_beta"],
         "p": r2["p_beta"], "r2": r2["r2"], "n": r2["n"]},
        {"model": "ret ~ 分段斜率", "term": "beta_pos", "coef": r3["coef"]["beta_pos(GEX>0)"],
         "t": r3["t"]["beta_pos(GEX>0)"], "p": r3["p"]["beta_pos(GEX>0)"],
         "r2": r3["r2"], "n": r3["n"]},
        {"model": "ret ~ 分段斜率", "term": "beta_neg", "coef": r3["coef"]["beta_neg(GEX<0)"],
         "t": r3["t"]["beta_neg(GEX<0)"], "p": r3["p"]["beta_neg(GEX<0)"],
         "r2": r3["r2"], "n": r3["n"]},
    ])
    regs.to_csv(run_dir / "gex_regressions.csv", index=False)

    # ---- 3) GEX 三档 ----
    df["tercile"] = pd.qcut(df["gex_yi_lag"], 3, labels=["低(最负)", "中", "高(最正)"])
    ter = df.groupby("tercile", observed=True).agg(
        n=("ret_bps", "size"), gex_min=("gex_yi_lag", "min"), gex_max=("gex_yi_lag", "max"),
        mean_bps=("ret_bps", "mean"), hit=("ret_bps", lambda s: (s > 0).mean()),
        sum_bps=("ret_bps", "sum")).reset_index()
    ter["sharpe"] = df.groupby("tercile", observed=True)["ret_bps"].agg(
        lambda s: s.mean() / s.std(ddof=1) * np.sqrt(252)).to_numpy()
    ter.to_csv(run_dir / "gex_terciles.csv", index=False)

    # ---- 4) 逐笔口径 ----
    pt_path = run_dir / "exc_per_trade.csv"
    pt_reg = None
    if pt_path.exists():
        pt = pd.read_csv(pt_path, parse_dates=["date"])
        pt = pt.merge(df[["date", "gex_yi_lag", "neg"]], on="date", how="inner")
        pt_reg = pt.groupby("neg").agg(
            n=("close_ret", "size"),
            close_ret_mean=("close_ret", "mean"),
            close_ret_hit=("close_ret", lambda s: (s > 0).mean()),
            mae_mean=("mae", "mean"),
            ratio_mean=("ratio", "mean"),
            actual_pnl_sum=("actual_pnl", "sum")).reset_index()
        pt_reg["regime"] = pt_reg["neg"].map({0: "GEX>0", 1: "GEX<0"})
        t_pt, p_pt = welch_t(pt.loc[pt.neg == 1, "close_ret"].to_numpy(),
                             pt.loc[pt.neg == 0, "close_ret"].to_numpy())
        pt_reg.to_csv(run_dir / "gex_per_trade_regime.csv", index=False)

    # ---- 5) 图 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = axes[0, 0]
    ax.fill_between(gex["date"], gex["gex_yi"], 0,
                    where=(gex["gex_yi"] >= 0), color="seagreen", alpha=0.5, label="GEX>0")
    ax.fill_between(gex["date"], gex["gex_yi"], 0,
                    where=(gex["gex_yi"] < 0), color="indianred", alpha=0.5, label="GEX<0")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("(a) 日度 GEX（亿元 / 1% 标的变动，300ETF 期权做市商口径）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.scatter(x[df.neg == 0], y[df.neg == 0], s=6, alpha=0.4, color="seagreen", label="GEX>0")
    ax.scatter(x[df.neg == 1], y[df.neg == 1], s=6, alpha=0.4, color="indianred", label="GEX<0")
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
    ax.plot(xs, r3["coef"]["const"] + np.clip(xs, 0, None) * r3["coef"]["beta_pos(GEX>0)"]
            + np.clip(xs, None, 0) * r3["coef"]["beta_neg(GEX<0)"],
            color="black", lw=1.5, label="分段拟合")
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel("GEX(t−1)（亿元/1%）")
    ax.set_ylabel("策略日收益 (bps)")
    ax.set_title("(b) GEX(t−1) vs 策略日收益")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1, 0]
    ax.bar(ter["tercile"].astype(str), ter["mean_bps"],
           color=["indianred", "steelblue", "seagreen"])
    for i, (v, n) in enumerate(zip(ter["mean_bps"], ter["n"])):
        ax.text(i, v, f"{v:.2f}\nn={n}", ha="center", va="bottom" if v > 0 else "top", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("(c) GEX 三档 → 平均日收益 (bps)")
    ax.grid(alpha=0.3)
    ax = axes[1, 1]
    if pt_reg is not None:
        ax.bar(pt_reg["regime"], pt_reg["close_ret_mean"] * 1e4,
               color=["seagreen", "indianred"])
        for i, v in enumerate(pt_reg["close_ret_mean"] * 1e4):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom" if v > 0 else "top", fontsize=9)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("(d) 逐笔收盘收益 (bps) × GEX 区间")
    ax.grid(alpha=0.3)
    fig.suptitle(f"GAMMA 状态 × 日内动量（{run_dir.name}，{len(df)} 日）")
    fig.tight_layout()
    fig.savefig(run_dir / "gex_analysis.png")
    plt.close(fig)

    # ---- 控制台汇总 ----
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n===== 1) 正/负 GEX 区间（t−1 对齐，策略日收益 bps） =====")
    print(regime.to_string(index=False))
    print(f"  均值差(负−正) Welch t = {t_diff:.2f}, p = {p_diff:.4f}")
    print("\n===== 2) 回归 =====")
    print(regs.to_string(index=False))
    print("\n===== 3) GEX 三档 =====")
    print(ter.to_string(index=False))
    if pt_reg is not None:
        print("\n===== 4) 逐笔口径 × GEX 区间 =====")
        print(pt_reg[["regime", "n", "close_ret_mean", "close_ret_hit", "mae_mean",
                      "ratio_mean", "actual_pnl_sum"]].to_string(index=False))
        print(f"  收盘收益均值差(负−正) Welch t = {t_pt:.2f}, p = {p_pt:.4f}")
    print(f"\n[gex] 产物 -> {run_dir}/gex_*.csv / .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
