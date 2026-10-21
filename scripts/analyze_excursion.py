#!/usr/bin/env python
"""持仓后行为研究（对指定 run，默认 A_g15_VM1_L14）：

1) 开仓后不利偏移 MAE（1 分钟 low/high 真实下探，占开仓价百分比）：
   平均值、百分位分位、频率分布（分桶）
2) 收盘收益 = (15:00 收盘/开仓价 − 1) × 方向；收益/回撤比 = 收盘收益 / MAE：
   分桶看规律（MAE 越大 → 收盘收益/比值如何变）
3) 日历效应（周一~周五）：交易数、平均 MAE、平均收盘收益、胜率、合计收益贡献
4) RSI 回归：日策略收益(bps) ~ 上一交易日 RSI(14) / RSI(5)（Wilder，标的日收盘）

输出（写入 run 目录，前缀 exc_）：
exc_per_trade.csv / exc_mae_distribution.csv / exc_weekday.csv /
exc_rsi_regression.csv / exc_analysis.png
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

from intramom.config import StrategyConfig
from intramom.metrics import ols_stats
from intramom.runner import build_panel

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
# MAE 分桶（% 名义）
BUCKETS = [0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, np.inf]
BUCKET_LABELS = ["0~0.1%", "0.1~0.2%", "0.2~0.3%", "0.3~0.5%", "0.5~0.75%",
                 "0.75~1%", "1~1.5%", "1.5~2%", "2~3%", ">3%"]
WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五"]


def rsi_wilder(close: pd.Series, n: int) -> pd.Series:
    """Wilder RSI（与前序 RUN-001 同口径）。"""
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    au = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return 100.0 - 100.0 / (1.0 + au / ad)


def per_trade_excursion(panel, cfg, rt, fills) -> pd.DataFrame:
    """逐整趟：MAE/MFE（1 分钟 low/high，自成交 bar 至 15:00）、收盘收益、收益/回撤比。"""
    idx = panel.idx
    i1500 = idx["15:00"]
    day_pos = {d: i for i, d in enumerate(panel.days)}
    rows = []
    for r in rt.itertuples():
        di = day_pos[r.date]
        op = fills[(fills.trade_id == r.trade_id) & fills.action.str.startswith("OPEN")].iloc[0]
        c0 = idx[op.exec_time]                     # 开仓成交 bar（开盘价成交）
        direction = 1 if r.direction == "long" else -1
        entry = r.entry_px
        lows = panel.L_m[di, c0:i1500 + 1]
        highs = panel.H_m[di, c0:i1500 + 1]
        if direction == 1:
            adverse = (np.nanmin(lows) / entry - 1.0)
            favor = (np.nanmax(highs) / entry - 1.0)
        else:
            adverse = -(np.nanmax(highs) / entry - 1.0)
            favor = -(np.nanmin(lows) / entry - 1.0)
        close_ret = (panel.C_m[di, i1500] / entry - 1.0) * direction
        mae = max(0.0, -adverse)
        mfe = max(0.0, favor)
        rows.append({
            "trade_id": r.trade_id, "date": r.date, "direction": r.direction,
            "entry_time": r.entry_time, "exec_bar": op.exec_time, "entry_px": entry,
            "weekday": pd.Timestamp(r.date).weekday(),
            "mae": mae, "mfe": mfe, "close_ret": close_ret,
            "ratio": (close_ret / mae) if mae > 0 else np.nan,
            "actual_ret_on_notional": r.ret_on_notional, "actual_pnl": r.pnl,
        })
    return pd.DataFrame(rows)


def mae_distribution(pt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MAE 百分位表 + 分桶频率表（含各桶收盘收益与收益/回撤比——看规律）。"""
    pct = pd.DataFrame({
        "percentile": PCTS,
        "mae": np.percentile(pt["mae"], PCTS),
        "mae_pct_of_entry": np.percentile(pt["mae"], PCTS) * 100,
    })
    pt_b = pt.copy()
    pt_b["bucket"] = pd.cut(pt_b["mae"], BUCKETS, labels=BUCKET_LABELS, right=False)
    freq = pt_b.groupby("bucket", observed=True).agg(
        n=("mae", "size"),
        share=("mae", lambda s: len(s) / len(pt_b)),
        mae_mean=("mae", "mean"),
        close_ret_mean=("close_ret", "mean"),
        close_ret_hit=("close_ret", lambda s: (s > 0).mean()),
        ratio_mean=("ratio", "mean"),
        ratio_median=("ratio", "median"),
    ).reset_index()
    return pct, freq


def weekday_table(pt: pd.DataFrame) -> pd.DataFrame:
    g = pt.groupby("weekday")
    df = g.agg(n=("mae", "size"),
               mae_mean=("mae", "mean"),
               mae_p90=("mae", lambda s: np.percentile(s, 90)),
               close_ret_mean=("close_ret", "mean"),
               close_ret_sum=("close_ret", "sum"),
               hit=("close_ret", lambda s: (s > 0).mean()),
               ratio_mean=("ratio", "mean"),
               actual_pnl_sum=("actual_pnl", "sum")).reset_index()
    df["weekday_cn"] = df["weekday"].map(lambda i: WEEKDAY_CN[i])
    return df


def rsi_regressions(panel, run_dir: Path) -> pd.DataFrame:
    """日策略收益(bps) ~ t−1 日 RSI(14)/RSI(5)（Wilder，标的日收盘）。"""
    eq = pd.read_csv(run_dir / "equity_daily.csv", parse_dates=["date"])
    close = panel._dl_full.set_index("trade_date")["close"]
    rows = []
    for n in (14, 5):
        rsi = rsi_wilder(close, n).shift(1)          # 上一交易日收盘可得
        rsi = rsi.reindex(eq["date"]).to_numpy(dtype=float)
        y_bps = eq["ret"].to_numpy(dtype=float) * 1e4
        st = ols_stats(y_bps, rsi)
        rows.append({"rsi_n": n, **st})
    return pd.DataFrame(rows)


def make_figure(pt, pct, freq, wk, title, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    # (a) MAE 直方图 + 分位线
    ax = axes[0, 0]
    ax.hist(pt["mae"] * 100, bins=60, color="steelblue", alpha=0.8)
    for p, c, lab in [(50, "black", "P50"), (90, "darkred", "P90"), (99, "red", "P99")]:
        v = pct.loc[pct.percentile == p, "mae"].iloc[0] * 100
        ax.axvline(v, color=c, ls="--", lw=1.2, label=f"{lab}={v:.2f}%")
    ax.set_title("(a) MAE 分布（开仓后最大不利偏移，% 开仓价）")
    ax.set_xlabel("MAE %")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # (b) MAE 分桶 → 平均收盘收益（规律检验）
    ax = axes[0, 1]
    x = np.arange(len(freq))
    ax.bar(x, freq["close_ret_mean"] * 1e4, color=["seagreen" if v > 0 else "indianred"
                                                   for v in freq["close_ret_mean"]])
    ax.set_xticks(x)
    ax.set_xticklabels(freq["bucket"], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    for i, (v, n) in enumerate(zip(freq["close_ret_mean"], freq["n"])):
        ax.text(i, v * 1e4, f"{v * 1e4:.1f}\nn={n}", ha="center",
                va="bottom" if v > 0 else "top", fontsize=7)
    ax.set_title("(b) MAE 分桶 → 平均收盘收益 (bps)")
    ax.grid(alpha=0.3)
    # (c) 星期效应
    ax = axes[1, 0]
    x = np.arange(len(wk))
    ax.bar(x, wk["close_ret_mean"] * 1e4, color="steelblue", alpha=0.85,
           label="平均收盘收益 (bps)")
    ax.set_xticks(x)
    ax.set_xticklabels(wk["weekday_cn"])
    ax.axhline(0, color="black", lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(x, wk["hit"], color="darkorange", marker="o", lw=1.2, label="胜率")
    ax2.set_ylabel("胜率")
    ax.set_title("(c) 星期效应：平均收盘收益与胜率")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    # (d) MAE × 收盘收益散点
    ax = axes[1, 1]
    ax.scatter(pt["mae"] * 100, pt["close_ret"] * 100, s=4, alpha=0.25, color="steelblue")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("MAE %")
    ax.set_ylabel("收盘收益 %")
    ax.set_title("(d) MAE vs 收盘收益（逐笔）")
    ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="持仓后行为研究：MAE 分布/收益回撤比/星期效应/RSI 回归")
    ap.add_argument("--run-dir",
                    default=str(ROOT / "data" / "results" / "sweeps" / "vm_L_g15_g05"
                                / "A_g15_VM1_L14_c1bp_s1t"))
    ap.add_argument("--out-prefix", default="exc")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = StrategyConfig.from_json(run_dir / "config_snapshot.json")
    rt = pd.read_csv(run_dir / "round_trips.csv", parse_dates=["date"])
    fills = pd.read_csv(run_dir / "fills.csv", parse_dates=["date"])
    print(f"[exc] run={run_dir.name} 整趟数={len(rt)}")
    panel = build_panel(cfg, ROOT)

    pt = per_trade_excursion(panel, cfg, rt, fills)
    pct, freq = mae_distribution(pt)
    wk = weekday_table(pt)
    rsi = rsi_regressions(panel, run_dir)

    title = (f"{run_dir.name}  持仓后行为研究 "
             f"(N={len(pt)}, 多头占 {(pt.direction == 'long').mean():.0%})")
    make_figure(pt, pct, freq, wk, title, run_dir / f"{args.out_prefix}_analysis.png")
    pt.to_csv(run_dir / f"{args.out_prefix}_per_trade.csv", index=False)
    pct.to_csv(run_dir / f"{args.out_prefix}_mae_percentiles.csv", index=False)
    freq.to_csv(run_dir / f"{args.out_prefix}_mae_buckets.csv", index=False)
    wk.to_csv(run_dir / f"{args.out_prefix}_weekday.csv", index=False)
    rsi.to_csv(run_dir / f"{args.out_prefix}_rsi_regression.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n===== 1) MAE（开仓后最大不利偏移，% 开仓价） =====")
    print(f"  平均 MAE = {pt['mae'].mean():.4%}  中位 = {pt['mae'].median():.4%}  "
          f"MAE=0 占比 = {(pt['mae'] == 0).mean():.2%}")
    print("  百分位:", {f"P{p}": f"{v:.3%}" for p, v in zip(PCTS, np.percentile(pt['mae'], PCTS))})
    print("\n  分桶频率（MAE 区间 → 占比 / 平均收盘收益 / 胜率 / 收益回撤比）:")
    f_show = freq.copy()
    f_show["share"] = f_show["share"].map("{:.1%}".format)
    f_show["close_ret_mean"] = f_show["close_ret_mean"].map("{:.3%}".format)
    f_show["mae_mean"] = f_show["mae_mean"].map("{:.3%}".format)
    f_show["close_ret_hit"] = f_show["close_ret_hit"].map("{:.1%}".format)
    print(f_show[["bucket", "n", "share", "mae_mean", "close_ret_mean",
                  "close_ret_hit", "ratio_mean", "ratio_median"]].to_string(index=False))

    print("\n===== 2) 收益/回撤比（收盘收益 / MAE） =====")
    r = pt["ratio"].dropna()
    print(f"  均值 = {r.mean():.3f}  中位 = {r.median():.3f}  "
          f"P25/P75 = {np.percentile(r, 25):.3f} / {np.percentile(r, 75):.3f}")
    print(f"  收盘收益均值 = {pt['close_ret'].mean():.4%}  胜率 = {(pt['close_ret'] > 0).mean():.2%}"
          f"（对照：引擎实际出场 平均净收益/名义 = {pt['actual_ret_on_notional'].mean():.4%}）")

    print("\n===== 3) 星期效应 =====")
    print(wk[["weekday_cn", "n", "mae_mean", "close_ret_mean", "hit", "ratio_mean",
              "actual_pnl_sum"]].to_string(index=False))

    print("\n===== 4) RSI 回归：日策略收益(bps) ~ t−1 RSI =====")
    for row in rsi.itertuples():
        print(f"  RSI({row.rsi_n}): α={row.alpha:.3f}bps β={row.beta:.4f} "
          f"t(β)={row.t_beta:.2f} p={row.p_beta:.4f} R²={row.r2:.4f} N={row.n}")
    print(f"\n[exc] 产物 -> {run_dir}/{args.out_prefix}_*.csv / .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
