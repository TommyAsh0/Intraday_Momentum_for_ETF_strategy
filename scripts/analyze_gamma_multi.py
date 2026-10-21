#!/usr/bin/env python
"""多标的 GAMMA 状态 × 日内动量：合并样本回归 + GEX 变体 + 中性区间剔除回测。

样本（各自 GEX 上市起）：
- 300etf → 510300 策略（A_g15_VM1_L14，2019-12 起，~1460 日）
- 500etf → 510500 策略（ext_510500_A_g15，2022-09 起，~800 日）
- cyb    → 159915 策略（ext_159915_A_g15，2022-09 起，~800 日）
合并后约 3,000 个（标的×日）观测，负 GEX 观测数约为单标的 3 倍。

分析：
1) 合并 + 分标的：正/负 GEX 区间收益对比（Welch t）、哑变量回归
2) GEX 变体对比：基准 / IV 加权 / 近月加权（区间分离度）
3) 合并三档（标的内分档，规避量纲差异）
4) GEX 中性区间剔除回测（300etf：中间档不交易 = 当日收益置 0，对比全指标）
5) 图：分标的区间柱 / 合并三档 / 变体对比 / 剔除净值

产物：data/results/analysis_gamma_multi/ 下 CSV 与 PNG。
"""
from __future__ import annotations

import math
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
from intramom.metrics import ols_stats, perf_stats
from intramom.options import UNDERLYINGS, load_option_data

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

OUT_DIR = ROOT / "data" / "results" / "analysis_gamma_multi"
PRED = ROOT.parent / "Intraday_Momentum_for_ETF" / "data" / "processed"

RUN_DIRS = {
    "300etf": ROOT / "data" / "results" / "sweeps" / "vm_L_g15_g05" / "A_g15_VM1_L14_c1bp_s1t",
    "500etf": ROOT / "data" / "results" / "ext_510500_A_g15",
    "cyb": ROOT / "data" / "results" / "ext_159915_A_g15",
}
STRAT_SYMBOL = {"300etf": "510300", "500etf": "510500", "cyb": "159915"}


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va / na + vb / nb)
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return t, p


def exdiv_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """由 fund_daily 的 pre_close 与昨收之差推除息（div_implied = max(0, 昨收 − pre_close)）。"""
    d = daily.sort_values("trade_date").copy()
    div = (d["close"].shift(1) - d["pre_close"]).clip(lower=0.0)
    out = d.loc[div > 1e-9, ["trade_date"]].copy()
    out["div_implied"] = div[div > 1e-9]
    return out.reset_index(drop=True)


def load_or_build_gex(key: str) -> pd.DataFrame:
    """读缓存的日度 GEX（含变体列）；缓存缺失或缺变体列则现算。"""
    cache = ROOT / "data" / "options" / f"gex_daily_{key}.csv"
    if cache.exists():
        g = pd.read_csv(cache, parse_dates=["date"])
        if "gex_ivw_yi" in g.columns:
            return g
    basic, daily = load_option_data(ROOT, key)
    code = UNDERLYINGS[key]["ts_code"].split(".")[0]
    if key == "300etf":
        etf = pd.read_csv(PRED / "daily_510300.csv", parse_dates=["trade_date"])
        exdiv = pd.read_csv(PRED / "exdiv_days_510300.csv")
    else:
        etf = pd.read_csv(ROOT / "data" / "processed_ext" / f"daily_{code}.csv",
                          parse_dates=["trade_date"])
        exdiv = exdiv_from_daily(etf)
    g, qa = build_gex_daily(basic, daily, etf, exdiv)
    g.to_csv(cache, index=False)
    print(f"[gex_multi] {key} GEX 面板: {qa['n_days']} 日, 负GEX {qa['gex_neg_days']} 日, "
          f"IV失败率 {qa['iv_fail_share']:.1%}")
    return g


def strategy_ret(key: str) -> pd.DataFrame:
    eq = pd.read_csv(RUN_DIRS[key] / "equity_daily.csv", parse_dates=["date"])
    return eq[["date", "ret"]]


def aligned_sample(key: str, gex_col: str = "gex_yi") -> pd.DataFrame:
    g = load_or_build_gex(key)
    df = strategy_ret(key).merge(g[["date", gex_col]], on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    df["gex_lag"] = df[gex_col].shift(1)
    df = df.dropna(subset=["gex_lag"]).reset_index(drop=True)
    df["ret_bps"] = df["ret"] * 1e4
    df["underlying"] = key
    return df


def regime_row(name: str, neg: np.ndarray, pos: np.ndarray) -> dict:
    t, p = welch_t(neg, pos)
    return {"sample": name, "n_neg": len(neg), "n_pos": len(pos),
            "mean_neg_bps": neg.mean(), "mean_pos_bps": pos.mean(),
            "diff_bps": neg.mean() - pos.mean(),
            "sharpe_neg": neg.mean() / neg.std(ddof=1) * np.sqrt(252),
            "sharpe_pos": pos.mean() / pos.std(ddof=1) * np.sqrt(252),
            "hit_neg": (neg > 0).mean(), "hit_pos": (pos > 0).mean(),
            "welch_t": t, "welch_p": p}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = {k: aligned_sample(k) for k in RUN_DIRS}
    pooled = pd.concat(samples.values(), ignore_index=True)
    pooled["neg"] = (pooled["gex_lag"] < 0).astype(int)
    print(f"[gex_multi] 合并样本: {len(pooled)} 观测，负GEX占 {pooled.neg.mean():.1%}")
    for k, s in samples.items():
        print(f"  {k}: {len(s)} 日, 负GEX {(s.gex_lag < 0).mean():.1%}, "
              f"GEX范围 [{s.gex_lag.min():.2f}, {s.gex_lag.max():.2f}] 亿")

    # ---- 1) 正/负区间：合并 + 分标的 ----
    rows = []
    for k, s in samples.items():
        rows.append(regime_row(k, s.loc[s.gex_lag < 0, "ret_bps"].to_numpy(),
                               s.loc[s.gex_lag >= 0, "ret_bps"].to_numpy()))
    rows.append(regime_row("POOLED", pooled.loc[pooled.neg == 1, "ret_bps"].to_numpy(),
                           pooled.loc[pooled.neg == 0, "ret_bps"].to_numpy()))
    regime = pd.DataFrame(rows)
    regime.to_csv(OUT_DIR / "multi_regime.csv", index=False)

    # 哑变量回归（合并 + 分标的）
    regs = []
    for name, s in list(samples.items()) + [("POOLED", pooled)]:
        neg_d = (s.gex_lag < 0).astype(float).to_numpy()
        r = ols_stats(s["ret_bps"].to_numpy(), neg_d)
        regs.append({"sample": name, "beta_neg_dummy": r["beta"], "t": r["t_beta"],
                     "p": r["p_beta"], "r2": r["r2"], "n": r["n"]})
    regs = pd.DataFrame(regs)
    regs.to_csv(OUT_DIR / "multi_dummy_regression.csv", index=False)

    # ---- 2) GEX 变体对比（区间分离度：负−正 均值差及其显著性） ----
    var_rows = []
    for col, lab in [("gex_yi", "基准(OI加权)"), ("gex_ivw_yi", "IV加权"),
                     ("gex_near_yi", "近月加权")]:
        ss = {k: aligned_sample(k, gex_col=col) for k in RUN_DIRS}
        for k, s in ss.items():
            var_rows.append(regime_row(f"{lab}|{k}",
                                       s.loc[s.gex_lag < 0, "ret_bps"].to_numpy(),
                                       s.loc[s.gex_lag >= 0, "ret_bps"].to_numpy()))
        sp = pd.concat(ss.values(), ignore_index=True)
        var_rows.append(regime_row(f"{lab}|POOLED",
                                   sp.loc[sp.gex_lag < 0, "ret_bps"].to_numpy(),
                                   sp.loc[sp.gex_lag >= 0, "ret_bps"].to_numpy()))
    variants = pd.DataFrame(var_rows)
    variants.to_csv(OUT_DIR / "multi_gex_variants.csv", index=False)

    # ---- 3) 合并三档（标的内分档） ----
    for k, s in samples.items():
        s["tercile"] = pd.qcut(s["gex_lag"], 3, labels=["低", "中", "高"])
    pooled_t = pd.concat(samples.values(), ignore_index=True)
    ter = pooled_t.groupby("tercile", observed=True).agg(
        n=("ret_bps", "size"), mean_bps=("ret_bps", "mean"),
        hit=("ret_bps", lambda x: (x > 0).mean()), sum_bps=("ret_bps", "sum")).reset_index()
    ter["sharpe"] = pooled_t.groupby("tercile", observed=True)["ret_bps"].agg(
        lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252)).to_numpy()
    ter.to_csv(OUT_DIR / "multi_terciles.csv", index=False)

    # ---- 4) GEX 中性区间剔除回测（300etf） ----
    s3 = samples["300etf"].copy()
    t1, t2 = s3["gex_lag"].quantile([1 / 3, 2 / 3])
    s3["ret_neutral_off"] = s3["ret"].where((s3.gex_lag < t1) | (s3.gex_lag > t2), 0.0)
    q1, q3 = s3["gex_lag"].quantile([0.25, 0.75])
    s3["ret_mid50_off"] = s3["ret"].where((s3.gex_lag < q1) | (s3.gex_lag > q3), 0.0)
    # 样本外版本：滚动 250 日分位阈值（当日决策只用 t−1 及以前的 GEX，无前视）
    roll_lo = s3["gex_lag"].rolling(250, min_periods=60).quantile(1 / 3)
    roll_hi = s3["gex_lag"].rolling(250, min_periods=60).quantile(2 / 3)
    s3["ret_neutral_oos"] = s3["ret"].where(
        roll_lo.isna() | (s3.gex_lag < roll_lo) | (s3.gex_lag > roll_hi), 0.0)
    init = 1_000_000.0
    excl_rows = []
    for col, lab in [("ret", "基准(全样本交易)"), ("ret_neutral_off", "剔除中间1/3(样本内)"),
                     ("ret_mid50_off", "剔除中间1/2(样本内)"),
                     ("ret_neutral_oos", "剔除中间1/3(滚动OOS)")]:
        r = s3[col].to_numpy()
        eq = init * np.cumprod(1.0 + r)
        st = perf_stats(r, eq, init=init)
        excl_rows.append({"variant": lab, **st})
    excl = pd.DataFrame(excl_rows)[["variant", "N", "TotalReturn", "IRR", "Vol",
                                    "Sharpe", "HitRatio", "MDD", "AvgDailyBps"]]
    excl.to_csv(OUT_DIR / "neutral_off_backtest.csv", index=False)
    s3[["date", "ret", "ret_neutral_off", "ret_mid50_off", "ret_neutral_oos",
        "gex_lag"]].to_csv(OUT_DIR / "neutral_off_daily.csv", index=False)

    # ---- 5) 图 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = axes[0, 0]
    x = np.arange(len(regime))
    ax.bar(x - 0.2, regime["mean_pos_bps"], width=0.4, color="seagreen", label="GEX>0")
    ax.bar(x + 0.2, regime["mean_neg_bps"], width=0.4, color="indianred", label="GEX<0")
    for i, r in regime.iterrows():
        ax.text(i, max(r.mean_pos_bps, r.mean_neg_bps), f"p={r.welch_p:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(regime["sample"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("(a) 正/负 GEX 区间日均收益 (bps)：分标的 + 合并")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.bar(ter["tercile"].astype(str), ter["mean_bps"],
           color=["indianred", "steelblue", "seagreen"])
    for i, v in enumerate(ter["mean_bps"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v > 0 else "top", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("(b) 合并三档（标的内分档）→ 平均日收益 (bps)")
    ax.grid(alpha=0.3)
    ax = axes[1, 0]
    vp = variants[variants["sample"].str.endswith("POOLED")]
    ax.bar(vp["sample"].str.split("|").str[0], vp["diff_bps"],
           color=["steelblue", "darkorange", "purple"])
    for i, (v, p) in enumerate(zip(vp["diff_bps"], vp["welch_p"])):
        ax.text(i, v, f"{v:.2f}\np={p:.3f}", ha="center", va="bottom" if v > 0 else "top",
                fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("(c) GEX 变体区间分离度（负−正 均值差, bps, 合并）")
    ax.grid(alpha=0.3)
    ax = axes[1, 1]
    ax.plot(s3["date"], np.cumprod(1 + s3["ret"]), lw=1.0, label="基准")
    ax.plot(s3["date"], np.cumprod(1 + s3["ret_neutral_off"]), lw=1.0, label="剔除中间1/3(样本内)")
    ax.plot(s3["date"], np.cumprod(1 + s3["ret_mid50_off"]), lw=1.0, label="剔除中间1/2(样本内)")
    ax.plot(s3["date"], np.cumprod(1 + s3["ret_neutral_oos"]), lw=1.2, ls="--",
            label="剔除中间1/3(滚动OOS)")
    ax.set_title("(d) GEX 中性区间剔除回测（300etf 净值）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("多标的 GAMMA 状态 × 日内动量（A 变体 g15 VM1 L14）")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multi_gex_analysis.png")
    plt.close(fig)

    # ---- 汇总 ----
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n===== 1) 正/负 GEX 区间（分标的 + 合并） =====")
    print(regime.to_string(index=False))
    print("\n===== 哑变量回归 ret~1[GEX<0] =====")
    print(regs.to_string(index=False))
    print("\n===== 2) GEX 变体（POOLED 行） =====")
    print(variants[variants["sample"].str.endswith("POOLED")].to_string(index=False))
    print("\n===== 3) 合并三档 =====")
    print(ter.to_string(index=False))
    print("\n===== 4) 中性区间剔除回测（300etf） =====")
    print(excl.to_string(index=False))
    print(f"\n[gex_multi] 产物 -> {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
