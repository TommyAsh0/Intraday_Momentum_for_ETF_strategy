#!/usr/bin/env python
"""挂单成交率与净值影响检验（原文参数口径）。

三种执行方式对照（标的 510300/510500，h3 + L-1 分钟止损 + band_state 重入，VM1/L14，万1）：
- open+slip0      ：次分钟开盘价成交、零滑点（原文/研报假设）
- open+slip1tick  ：次分钟开盘价成交 + 每份 1 tick 滑点成本（我们先前的保守假设）
- passive_chase   ：限价挂单（买=open−1tick / 卖=open+1tick），当根 bar 触及即成交于限价
                    （比市价省 1 tick），未触及则以该 bar 收盘追价成交；15:00 平仓走收盘价

产物：data/results/article_check/fill_model_check.csv + 控制台对照。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from intramom import load_base_config, build_panel  # noqa: E402
from intramom.engine import run_backtest  # noqa: E402
from intramom.ext_loader import load_data_ext  # noqa: E402
from intramom.metrics import perf_stats  # noqa: E402
from intramom.panel import PanelData  # noqa: E402

OUT = ROOT / "data" / "results" / "article_check"
INIT = 1_000_000.0
TICK = 0.001


def fill_stats(panel: PanelData, trades: pd.DataFrame) -> dict:
    """逐笔重判被动成交/追价/收盘平仓；追价的平均不利偏移（tick 数）。"""
    idx = panel.idx
    day_pos = {d: i for i, d in enumerate(panel.days)}
    n_pass = n_chase = n_eod = 0
    chase_cost_ticks = []
    for r in trades.itertuples():
        if r.decision_time == "15:00":
            n_eod += 1
            continue
        di = day_pos[pd.Timestamp(r.date)]
        ec = idx[r.exec_time]
        o, h, l, c = (panel.O_m[di, ec], panel.H_m[di, ec],
                      panel.L_m[di, ec], panel.C_m[di, ec])
        if r.side == "买入":
            lim = round(o - TICK, 3)
            ok = l <= lim
            if not ok:
                chase_cost_ticks.append((c - o) / TICK)
        else:
            lim = round(o + TICK, 3)
            ok = h >= lim
            if not ok:
                chase_cost_ticks.append((o - c) / TICK)
        n_pass += int(ok)
        n_chase += int(not ok)
    n_intra = n_pass + n_chase
    return {"fill_rate": n_pass / max(n_intra, 1), "n_pass": n_pass, "n_chase": n_chase,
            "n_eod": n_eod,
            "chase_avg_ticks": float(np.mean(chase_cost_ticks)) if chase_cost_ticks else 0.0}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_base_config(ROOT).with_overrides(
        grid="custom", grid_points=("10:29", "11:29", "13:59"), stop_freq="1min",
        entry_mode="band_state", vol_mult=1.0, lookback=14, vol_source="index")

    panels = {"510300.SH": None, "510500.SH": None}
    panels["510300.SH"] = build_panel(base, ROOT)
    bundle, _ = load_data_ext("510500.SH", base.start, base.end, ROOT)
    panels["510500.SH"] = PanelData.build(bundle)

    rows = []
    for code in panels:
        for cname, cp in {"A": {"stop_mode": "opp_band", "sizing": "fixed"},
                          "B": {"stop_mode": "band_vwap", "sizing": "fixed"},
                          "C": {"stop_mode": "band_vwap", "sizing": "vol_target"}}.items():
            res0 = res1 = resp = None
            for fm, slip, lab in [("open", 0.0, "市价+0滑点(原文)"),
                                  ("open", 0.001, "市价+1tick滑点"),
                                  ("passive_chase", 0.0, "挂单被动+追价")]:
                cfg = base.with_overrides(symbol=code, fill_model=fm,
                                          slippage_per_share=slip, **cp)
                res = run_backtest(panels[code], cfg)
                r = res.daily["ret"].to_numpy()
                eq = INIT * np.cumprod(1 + r)
                st = perf_stats(r, eq, init=INIT)
                fs = fill_stats(panels[code], res.trades) if fm == "passive_chase" else {}
                rows.append({"code": code, "配置": cname, "执行": lab,
                             "TotalReturn": st["TotalReturn"], "IRR": st["IRR"],
                             "Sharpe": st["Sharpe"], "MDD": st["MDD"],
                             "exec": len(res.trades), **fs})
                if fm == "passive_chase":
                    resp = res
            # 净值比：挂单 vs 市价（0滑点）
            rows[-1]["净值比_挂单/市价0滑点"] = rows[-1]["TotalReturn"] / max(
                rows[-3]["TotalReturn"], 1e-12)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fill_model_check.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    for code in panels:
        print(f"\n===== {code}（原文参数：band_state + h3 + L-1 + VM1/L14 + 万1）=====")
        sub = df[df.code == code][["配置", "执行", "TotalReturn", "IRR", "Sharpe", "MDD",
                                   "fill_rate", "chase_avg_ticks"]]
        print(sub.to_string(index=False))
    print(f"\n[fill] 产物 -> {OUT}/fill_model_check.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
