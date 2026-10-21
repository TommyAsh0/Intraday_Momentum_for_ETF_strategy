#!/usr/bin/env python
"""胜/败日连续性与集聚检验（对指定 run 的日收益序列）。

问题：噪音日（败）与趋势日（胜）会连续吗？会集中在某些时段吗？
方法：
1) 马尔可夫转移：P(胜|连胜) vs P(胜|连负)，Fisher 精确检验
2) Wald–Wolfowitz 游程检验（胜/败序列随机性）
3) ACF：ret、|ret|（波动集聚）、胜败指示（lags 1-10，Bartlett 95% 界）
4) 连胜/连败长度分布 vs 几何分布零假设
5) 60 日滚动胜率/收益时序（时段集聚可视化）
6) GEX 自相关（佐证 regime 持续性，若 GEX 面板存在）

产物（写入 run 目录，前缀 cls_）：cls_transitions.csv / cls_runs_test.csv /
cls_acf.csv / cls_streaks.csv / cls_analysis.png
"""
from __future__ import annotations

import argparse
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

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]


def runs_test(binary: np.ndarray) -> dict:
    """Wald–Wolfowitz 游程检验。binary: 0/1 序列。"""
    x = np.asarray(binary)
    n1, n2 = int(x.sum()), int(len(x) - x.sum())
    runs = int(1 + (x[1:] != x[:-1]).sum())
    mu = 2.0 * n1 * n2 / (n1 + n2) + 1.0
    var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)
           / ((n1 + n2) ** 2 * (n1 + n2 - 1)))
    z = (runs - mu) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"n_win": n1, "n_loss": n2, "n_runs": runs, "expected_runs": mu,
            "z": z, "p": p}


def transition_test(win: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """转移矩阵 + Fisher 精确检验（P(胜|连胜) 是否显著不同于 P(胜|连负)）。"""
    from scipy.stats import fisher_exact
    prev, cur = win[:-1], win[1:]
    ww = int((prev & cur).sum())
    wl = int((prev & ~cur).sum())
    lw = int((~prev & cur).sum())
    ll = int((~prev & ~cur).sum())
    tbl = pd.DataFrame({"次日胜": [ww, lw], "次日负": [wl, ll]},
                       index=["当日胜", "当日负"])
    p_win_after_win = ww / max(ww + wl, 1)
    p_win_after_loss = lw / max(lw + ll, 1)
    odds, p = fisher_exact([[ww, wl], [lw, ll]])
    stats = {"P(胜|连胜)": p_win_after_win, "P(胜|连负)": p_win_after_loss,
             "diff": p_win_after_win - p_win_after_loss,
             "odds_ratio": odds, "fisher_p": p}
    return tbl, stats


def acf(x: np.ndarray, lags: int = 10) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = (x * x).sum()
    return np.array([1.0] + [(x[k:] * x[:-k]).sum() / denom for k in range(1, lags + 1)])


def streaks(win: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """连胜/连败长度分布 + 与几何零假设对比。"""
    lengths, cur = [], 1
    for i in range(1, len(win)):
        if win[i] == win[i - 1]:
            cur += 1
        else:
            lengths.append((bool(win[i - 1]), cur))
            cur = 1
    lengths.append((bool(win[-1]), cur))
    df = pd.DataFrame(lengths, columns=["is_win", "length"])
    p_win = win.mean()
    summ = {}
    for is_w, name in [(True, "win"), (False, "loss")]:
        L = df.loc[df.is_win == is_w, "length"].to_numpy()
        p = p_win if is_w else 1 - p_win
        # 几何零假设下最长 streak 的期望近似 E[max] ≈ log_{1/p}(n·(1-p)+p)/-ln... 用模拟更稳
        rng = np.random.default_rng(7)
        sim_max = []
        n = len(win)
        for _ in range(2000):
            s = rng.random(n) < p
            mx = c = 1
            for i in range(1, n):
                c = c + 1 if s[i] == s[i - 1] else 1
                mx = max(mx, c)
            sim_max.append(mx)
        summ[name] = {"longest": int(L.max()), "mean_len": float(L.mean()),
                      "null_max_p95": float(np.percentile(sim_max, 95)),
                      "n_streaks": int(len(L))}
    return df, summ


def main() -> int:
    ap = argparse.ArgumentParser(description="胜/败日连续性与集聚检验")
    ap.add_argument("--run-dir",
                    default=str(ROOT / "data" / "results" / "sweeps" / "vm_L_g15_g05"
                                / "A_g15_VM1_L14_c1bp_s1t"))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    eq = pd.read_csv(run_dir / "equity_daily.csv", parse_dates=["date"])
    ret = eq["ret"].to_numpy(float)
    active = ret != 0                       # 剔除无交易日（平）
    win = (ret[active] > 0)
    dates = eq["date"].to_numpy()[active]
    ret_a = ret[active]
    print(f"[cls] {run_dir.name}: 全样本 {len(ret)} 日，活跃（有盈亏）{active.sum()} 日，"
          f"胜率 {win.mean():.2%}")

    tbl, tr_stats = transition_test(win)
    rt = runs_test(win.astype(int))
    lags = 10
    acf_ret = acf(ret_a, lags)
    acf_abs = acf(np.abs(ret_a), lags)
    acf_win = acf(win.astype(float), lags)
    band = 1.96 / math.sqrt(len(win))
    acf_df = pd.DataFrame({"lag": range(lags + 1), "acf_ret": acf_ret,
                           "acf_abs_ret": acf_abs, "acf_win": acf_win,
                           "bartlett_95": band})
    st_df, st_stats = streaks(win)

    tbl.to_csv(run_dir / "cls_transitions.csv")
    pd.DataFrame([tr_stats]).to_csv(run_dir / "cls_transition_stats.csv", index=False)
    pd.DataFrame([rt]).to_csv(run_dir / "cls_runs_test.csv", index=False)
    acf_df.to_csv(run_dir / "cls_acf.csv", index=False)
    st_df.to_csv(run_dir / "cls_streaks.csv", index=False)

    # ---- 图 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    s_hit = pd.Series(win.astype(float)).rolling(60).mean()
    s_ret = pd.Series(ret_a).rolling(60).mean() * 1e4
    ax.plot(dates, s_hit, color="steelblue", lw=1.2, label="60日滚动胜率")
    ax.axhline(win.mean(), color="gray", ls="--", lw=0.8, label=f"全样本胜率 {win.mean():.1%}")
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(dates, s_ret, color="darkorange", lw=1.0, alpha=0.8, label="60日滚动均值 (bps)")
    ax2.axhline(0, color="black", lw=0.6)
    ax.set_title("(a) 60 日滚动胜率与收益（时段集聚）")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    ax = axes[0, 1]
    x = np.arange(1, lags + 1)
    w = 0.27
    ax.bar(x - w, acf_ret[1:], width=w, label="ret", color="steelblue")
    ax.bar(x, acf_abs[1:], width=w, label="|ret|", color="darkorange")
    ax.bar(x + w, acf_win[1:], width=w, label="胜败指示", color="seagreen")
    ax.axhline(band, color="red", ls="--", lw=0.8)
    ax.axhline(-band, color="red", ls="--", lw=0.8, label="95% 界")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_title("(b) 自相关 ACF（lags 1-10）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1, 0]
    for is_w, name, c in [(True, "连胜", "seagreen"), (False, "连败", "indianred")]:
        L = st_df.loc[st_df.is_win == is_w, "length"]
        vc = L.value_counts().sort_index()
        ax.plot(vc.index, vc.values, "o-", color=c, label=f"{name} (max={L.max()})")
    ax.set_yscale("log")
    ax.set_xlabel("连续长度（日）")
    ax.set_ylabel("频次（log）")
    ax.set_title("(c) 连胜/连败长度分布")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1, 1]
    labels = ["P(胜|连胜)", "P(胜|连负)", "全样本胜率"]
    vals = [tr_stats["P(胜|连胜)"], tr_stats["P(胜|连负)"], win.mean()]
    ax.bar(labels, vals, color=["seagreen", "indianred", "gray"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title(f"(d) 转移概率（Fisher p={tr_stats['fisher_p']:.4f}）")
    ax.grid(alpha=0.3)
    fig.suptitle(f"胜/败日集聚检验 — {run_dir.name}")
    fig.tight_layout()
    fig.savefig(run_dir / "cls_analysis.png")
    plt.close(fig)

    # ---- 汇总 ----
    print("\n===== 1) 转移检验 =====")
    print(tbl.to_string())
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in tr_stats.items()})
    print("\n===== 2) 游程检验 =====")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rt.items()})
    print("\n===== 3) ACF（|·|>95%界为显著） =====")
    print(acf_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n===== 4) 连胜/连败 =====")
    print(st_stats)
    print(f"\n[cls] 产物 -> {run_dir}/cls_*.csv / .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
