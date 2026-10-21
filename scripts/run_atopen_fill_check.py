#!/usr/bin/env python
"""开仓/平仓按"开盘价挂单"的成交率与净值影响检验。

模型：决策后以**次分钟 bar 的开盘价本身**挂限价单（买 limit=open，卖 limit=open）。
bar 级判定（1 分钟 OHLC 所能给出的最细口径）：
- 穿价成交：买 low < open / 卖 high > open —— 价格真穿过限价，必成交
- 触价：low == open / high == open —— 只有开盘一笔触价，排队不确定性，给两种情景
- 未成交（含触价未成交情景）→ 以该 bar 收盘价追价成交
15:00 平仓仍走收盘价。

净值影响通过逐笔重算实现精确测算（改成交价→改当趟盈亏→逐日 AUM 递推），
成交决策不依赖成交价，故无须重跑引擎。

产物：data/results/article_check/at_open_fill_check.csv + 控制台对照。
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


def judge_at_open(panel: PanelData, trades: pd.DataFrame) -> pd.DataFrame:
    """逐笔判定：穿价/触价（对开盘价挂单而言不可能 low>open，故不存在"完全不触及"）。"""
    idx = panel.idx
    day_pos = {d: i for i, d in enumerate(panel.days)}
    rows = []
    for r in trades.itertuples():
        if r.decision_time == "15:00":
            rows.append({"trade_id": r.trade_id, "date": r.date, "kind": "EOD"})
            continue
        di = day_pos[pd.Timestamp(r.date)]
        ec = idx[r.exec_time]
        o, h, l, c = (panel.O_m[di, ec], panel.H_m[di, ec],
                      panel.L_m[di, ec], panel.C_m[di, ec])
        if r.side == "买入":
            strict, touch = l < o, l == o
        else:
            strict, touch = h > o, h == o
        rows.append({"trade_id": r.trade_id, "date": r.date, "side": r.side,
                     "action": r.action, "exec_time": r.exec_time,
                     "kind": "strict" if strict else ("touch" if touch else "none"),
                     "open": o, "close": c, "price": r.price, "shares": r.shares,
                     "commission_rate": r.commission / (r.price * r.shares)})
    return pd.DataFrame(rows)


def rebuild_nav(base_daily: pd.DataFrame, judge: pd.DataFrame,
                touch_fills: bool) -> tuple[np.ndarray, float]:
    """情景重算净值：touch_fills=True 触价视为成交（乐观）；False 触价追价到收盘（保守）。

    基线 = 市价 open 成交（slip0）；调整 = 触价未成交的笔改按收盘成交。
    仅改成交价不改决策，逐趟盈亏差 → 逐日 pnl 差 → AUM 精确递推。
    """
    # 每笔成交价的调整量（相对基线 open 价）
    adj = judge[(judge.kind == "touch") & (judge.action != "")].copy()
    if touch_fills:
        adj = adj.iloc[0:0]
    adj["delta"] = np.where(adj["side"] == "买入",
                            -(adj["close"] - adj["open"]) * adj["shares"],
                            +(adj["close"] - adj["open"]) * adj["shares"])
    # 佣金差（成交价变动引起）： commission = px*shares*rate
    adj["delta"] -= np.where(adj["side"] == "买入",
                             (adj["close"] - adj["open"]) * adj["shares"] * adj["commission_rate"],
                             (adj["open"] - adj["close"]) * adj["shares"] * adj["commission_rate"])
    day_delta = adj.groupby("date")["delta"].sum()
    d = base_daily.copy()
    d["delta"] = d["date"].map(day_delta).fillna(0.0)
    aum = INIT
    rets = []
    for r in d.itertuples():
        pnl = r.ret * r.aum_start + r.delta
        rets.append(pnl / aum)
        aum += pnl
    return np.array(rets), aum


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_base_config(ROOT).with_overrides(
        grid="custom", grid_points=("10:29", "11:29", "13:59"), stop_freq="1min",
        entry_mode="band_state", vol_mult=1.0, lookback=14, vol_source="index",
        slippage_per_share=0.0)

    panels = {"510300.SH": build_panel(base, ROOT)}
    bundle, _ = load_data_ext("510500.SH", base.start, base.end, ROOT)
    panels["510500.SH"] = PanelData.build(bundle)

    rows = []
    for code, panel in panels.items():
        for cname, cp in {"A": {"stop_mode": "opp_band", "sizing": "fixed"},
                          "B": {"stop_mode": "band_vwap", "sizing": "fixed"},
                          "C": {"stop_mode": "band_vwap", "sizing": "vol_target"}}.items():
            cfg = base.with_overrides(symbol=code, **cp)
            res = run_backtest(panel, cfg)
            j = judge_at_open(panel, res.trades)
            intra = j[j.kind != "EOD"]
            n_strict = int((intra.kind == "strict").sum())
            n_touch = int((intra.kind == "touch").sum())
            n_eod = int((j.kind == "EOD").sum())
            # 分解：开仓 vs 止损平仓
            op = intra[intra.action.str.startswith("OPEN")]
            cl = intra[intra.action.str.startswith("CLOSE")]
            fr_open = (op.kind == "strict").mean()
            fr_stop = (cl.kind == "strict").mean()
            # 净值情景
            r0 = res.daily["ret"].to_numpy()
            eq0 = INIT * np.cumprod(1 + r0)
            st0 = perf_stats(r0, eq0, init=INIT)
            r_opt, aum_opt = rebuild_nav(res.daily, j, touch_fills=True)
            st_opt = perf_stats(r_opt, INIT * np.cumprod(1 + r_opt), init=INIT)
            r_pes, aum_pes = rebuild_nav(res.daily, j, touch_fills=False)
            st_pes = perf_stats(r_pes, INIT * np.cumprod(1 + r_pes), init=INIT)
            rows.append({"code": code, "配置": cname,
                         "穿价成交率(下限)": n_strict / max(len(intra), 1),
                         "触价占比": n_touch / max(len(intra), 1),
                         "有效成交率估计(穿+0.5触)": (n_strict + 0.5 * n_touch) / max(len(intra), 1),
                         "开仓穿价率": fr_open, "止损穿价率": fr_stop,
                         "n_strict": n_strict, "n_touch": n_touch, "n_eod": n_eod,
                         "TR_全成交": st0["TotalReturn"], "Sharpe_全成交": st0["Sharpe"],
                         "TR_触价成交": st_opt["TotalReturn"], "Sharpe_触价成交": st_opt["Sharpe"],
                         "TR_触价追价": st_pes["TotalReturn"], "Sharpe_触价追价": st_pes["Sharpe"],
                         "净值差_保守vs全成交": st_pes["TotalReturn"] - st0["TotalReturn"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "at_open_fill_check.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    for code in panels:
        print(f"\n===== {code}（原文参数：开盘价挂单）=====")
        print(df[df.code == code].to_string(index=False))
    print(f"\n[at_open] 产物 -> {OUT}/at_open_fill_check.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
