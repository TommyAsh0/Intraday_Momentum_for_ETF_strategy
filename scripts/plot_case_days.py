#!/usr/bin/env python
"""案例日图：从某个 run 目录挑亏损整趟，画论文 Fig2 风格日内图（价格 + 1 分钟 UB/LB 噪声带
+ VWAP 步进线 + 成交标记），并输出具体成交时间与指标带宽。

用法：
    # 变体 B 最亏的 3 个案例（默认 run = base_B_g30）
    python scripts/plot_case_days.py
    # 指定日期 / 指定 run / 亏损最多 N 个
    python scripts/plot_case_days.py --run-dir data/results/base_A_g30 --dates 2022-04-29
    python scripts/plot_case_days.py --run-dir data/results/sweeps/vm_L_g15_g05_B/B_g15_VM2_L20_c1bp_s1t -n 4

输出（写入 run 目录）：case_loss_days.png、case_loss_days.csv（案例明细：时间/价格/σ/UB/LB/带宽/盈亏）。
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
from intramom.data_loader import SIG_BAR
from intramom.runner import build_panel

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

TICKS = ["09:31", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00"]


def intraday_axis(ax, ppos):
    ax.set_xticks([ppos[t] for t in TICKS])
    ax.set_xticklabels(TICKS, fontsize=8)
    ax.axvline(ppos["11:30"] + 0.5, color="gray", ls=":", lw=1.0)
    ax.set_xlim(-2, 241)
    ax.set_xlabel("Time of day (lunch break removed; dotted line = 11:30 | 13:01)", fontsize=8)


def plot_case(ax, di, panel, cfg, fills_d, rt_row, title):
    """画单个案例日：价格/UB/LB/噪声区/VWAP/成交标记 + 信息框。"""
    ppos, idx = panel.ppos, panel.idx
    xmin = np.arange(len(panel.price_labels))
    close = panel.C_m[di, 1:]
    ub1m, lb1m = panel.bands(di, cfg.vol_mult, cfg.lookback)
    ax.plot(xmin, close, color="black", lw=1.2, label=f"{cfg.symbol} price")
    ax.plot(xmin, ub1m, color="tab:red", lw=1.0, label="UB (1-min)")
    ax.plot(xmin, lb1m, color="tab:green", lw=1.0, label="LB (1-min)")
    ax.fill_between(xmin, lb1m, ub1m, alpha=0.15, color="gray", label="Noise area")
    # VWAP（该 run 网格口径）步进线
    taus = cfg.grid_taus
    vw = panel.vwap_panel(taus)
    sigs = [SIG_BAR.get(t, t) for t in taus] + ["15:00"]
    xv = [ppos[s] for s in sigs]
    vv = [vw[di, idx[s]] for s in sigs]
    ax.step(xv, vv, where="post", color="tab:blue", lw=1.0, ls="--",
            label=f"VWAP ({cfg.grid} grid)")
    # 成交标记（▲买/▼卖）+ 时间标注
    for k, r in enumerate(fills_d.itertuples()):
        x = ppos[r.exec_time]
        mk = "^" if r.side == "买入" else "v"
        c = "tab:red" if r.side == "买入" else "tab:green"
        ax.scatter([x], [r.price], marker=mk, color=c, s=60, zorder=5)
        dy = 14 if (k % 2 == 0) else -18
        ax.annotate(f"{r.exec_time}\n{r.side} {r.price:.3f}", (x, r.price),
                    textcoords="offset points", xytext=(0, dy), ha="center",
                    fontsize=7, color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6, alpha=0.6))
    ax.set_title(title, fontsize=10)
    intraday_axis(ax, ppos)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.25)


def case_metrics(di, panel, cfg, rt_row, fills_d):
    """案例明细：入场/出场时点的 σ/UB/LB/VWAP 与带宽（引擎同口径）。"""
    ppos = panel.ppos
    ub1m, lb1m = panel.bands(di, cfg.vol_mult, cfg.lookback)
    sig = panel.sigma_panel(cfg.lookback)[di]
    taus = cfg.grid_taus
    vw = panel.vwap_panel(taus)[di] if cfg.stop_mode in ("band_vwap", "vwap_only") else None

    def at(tau):
        sbar = SIG_BAR.get(tau, tau)
        pi = ppos[sbar]
        sc = panel.idx[sbar]
        ub, lb, sg = ub1m[pi], lb1m[pi], sig[pi]
        v = vw[sc] if vw is not None else np.nan
        stop = np.nan
        if cfg.stop_mode == "band_vwap":
            stop = max(ub, v) if rt_row["direction"] == "long" else min(lb, v)
        elif cfg.stop_mode == "opp_band":
            stop = lb if rt_row["direction"] == "long" else ub
        elif cfg.stop_mode == "vwap_only":
            stop = v
        return sg, ub, lb, v, stop

    sg_e, ub_e, lb_e, vw_e, stop_e = at(rt_row["entry_time"])
    sg_x, ub_x, lb_x, vw_x, stop_x = at(rt_row["exit_time"])
    px = rt_row["entry_px"]
    return {
        "date": str(pd.Timestamp(rt_row["date"]).date()),
        "direction": rt_row["direction"],
        "entry_decision": rt_row["entry_time"],
        "entry_exec": fills_d[fills_d.action.str.startswith("OPEN")].iloc[0]["exec_time"],
        "entry_px": px,
        "exit_decision": rt_row["exit_time"],
        "exit_exec": fills_d[fills_d.action.str.startswith("CLOSE")].iloc[-1]["exec_time"],
        "exit_px": rt_row["exit_px"],
        "shares": rt_row["shares"],
        "sigma_entry": sg_e, "UB_entry": ub_e, "LB_entry": lb_e, "VWAP_entry": vw_e,
        "bandwidth_entry": ub_e - lb_e, "bandwidth_entry_pct": (ub_e - lb_e) / px,
        "sigma_exit": sg_x, "UB_exit": ub_x, "LB_exit": lb_x, "VWAP_exit": vw_x,
        "bandwidth_exit": ub_x - lb_x, "bandwidth_exit_pct": (ub_x - lb_x) / rt_row["exit_px"],
        "stop_at_exit": stop_x,
        "gross": rt_row["gross"], "costs": rt_row["costs"], "pnl": rt_row["pnl"],
        "ret_on_notional": rt_row["ret_on_notional"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="亏损案例日图（Fig2 风格）")
    ap.add_argument("--run-dir", default=str(ROOT / "data" / "results" / "base_B_g30"))
    ap.add_argument("--dates", nargs="*", default=None, help="指定案例日期 YYYY-MM-DD（默认取最亏 N 趟）")
    ap.add_argument("-n", type=int, default=3, help="最亏案例数（默认 3）")
    ap.add_argument("--out-name", default="case_loss_days")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = StrategyConfig.from_json(run_dir / "config_snapshot.json")
    rt = pd.read_csv(run_dir / "round_trips.csv", parse_dates=["date"])
    fills = pd.read_csv(run_dir / "fills.csv", parse_dates=["date"])

    if args.dates:
        cases = rt[rt.date.isin(pd.to_datetime(args.dates))]
        if cases.empty:
            raise SystemExit(f"指定日期在该 run 无整趟: {args.dates}")
    else:
        cases = rt.nsmallest(args.n, "ret_on_notional")
    cases = cases.sort_values("date")

    print(f"[plot_case] run={run_dir.name} 变体={cfg.variant_tag} 网格={cfg.grid} "
          f"VM={cfg.vol_mult:g} L={cfg.lookback} → {len(cases)} 个亏损案例")
    panel = build_panel(cfg, ROOT)
    day_pos = {d: i for i, d in enumerate(panel.days)}

    fig, axes = plt.subplots(1, len(cases), figsize=(6.2 * len(cases), 4.6), squeeze=False)
    rows = []
    for ax, (_, r) in zip(axes[0], cases.iterrows()):
        di = day_pos[r["date"]]
        fills_d = fills[fills.trade_id == r["trade_id"]]
        m = case_metrics(di, panel, cfg, r, fills_d)
        rows.append(m)
        dlab = "多" if m["direction"] == "long" else "空"
        title = (f"变体{cfg.variant_tag} 亏损案例 {m['date']} ({dlab})  "
                 f"pnl={m['pnl']:,.0f} ({m['ret_on_notional']:.2%})")
        plot_case(ax, di, panel, cfg, fills_d, r, title)
        print(f"\n--- {m['date']} {dlab} (trade_id={r['trade_id']}) ---")
        print(f"  入场: 决策 {m['entry_decision']} → 成交 {m['entry_exec']} @ {m['entry_px']:.3f}"
              f" × {int(m['shares'])} 份")
        print(f"    σ={m['sigma_entry']:.4%}  UB={m['UB_entry']:.4f}  LB={m['LB_entry']:.4f}"
              f"  VWAP={m['VWAP_entry']:.4f}")
        print(f"    带宽 = {m['bandwidth_entry']:.4f} 元 = {m['bandwidth_entry_pct']:.2%} 名义")
        print(f"  出场: 决策 {m['exit_decision']} → 成交 {m['exit_exec']} @ {m['exit_px']:.3f}"
              f"  止损位={m['stop_at_exit']:.4f}")
        print(f"    σ={m['sigma_exit']:.4%}  UB={m['UB_exit']:.4f}  LB={m['LB_exit']:.4f}"
              f"  VWAP={m['VWAP_exit']:.4f}")
        print(f"    带宽 = {m['bandwidth_exit']:.4f} 元 = {m['bandwidth_exit_pct']:.2%} 名义")
        print(f"  盈亏: gross={m['gross']:,.0f}  costs={m['costs']:,.0f}  pnl={m['pnl']:,.0f}"
              f" ({m['ret_on_notional']:.2%})")

    fig.tight_layout()
    png = run_dir / f"{args.out_name}.png"
    fig.savefig(png)
    plt.close(fig)
    csv = run_dir / f"{args.out_name}.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"\n[plot_case] 图 -> {png}\n[plot_case] 明细 -> {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
