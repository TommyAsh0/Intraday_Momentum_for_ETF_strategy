#!/usr/bin/env python
"""西部证券《指数化配置系列研究：确定趋势，ETF 日内动量策略》(2025-12-17) 改进链复现。

改进链（研报原文口径，逐步累加）：
- S0 原始策略：VM=1/L=14 噪声带+昨收修正；10:29/11:29/13:59 判断、次分钟开盘成交；
  对向边界每分钟止损；带外即入（band_state）；万1 无滑点
- S1 +5minVWAP：成交价改信号后 5 分钟 VWAP（EOD=最后 5 分钟 VWAP）
- S2 +放松平仓：多 min(UB,VWAP) / 空 max(LB,VWAP)（band_vwap_loose）
- S3 +收窄退出边界 m=0.7（入场带不变，退出带 ×0.7）
- S4 +阶梯止盈：浮盈 > stop_rate（2024 前指数平均日内收益绝对值）先平 50%
另：ETF 底仓增强 = 50% B&H + 50% 现金每日只执行第一个信号的一趟交易。

标的：510300/510500/159845 ETF + 000905 指数（研报主线）。样本 2017-01 ~ 2025-10-10（对齐研报）。
产物：data/results/wb_reproduction/{chain_summary.csv, overlay_summary.csv,
yearly_<code>.csv, chain_<code>.png} + reports/wb_report_reproduction.md 的数据基础。
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

from intramom.config import load_base_config
from intramom.data_loader import load_data
from intramom.engine import run_backtest
from intramom.etf_loader import load_data_etf
from intramom.ext_loader import load_data_ext
from intramom.index_loader import load_data_index, load_index_daily_df
from intramom.metrics import perf_stats, round_trips
from intramom.panel import PanelData

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

OUT = ROOT / "data" / "results" / "wb_reproduction"
INIT = 1_000_000.0
SAMPLE_END = "2025-10-10"   # 研报样本终点
STOPRATE_END = pd.Timestamp("2025-01-01")  # stop_rate 用 2024-12-31 之前

INSTRUMENTS = [("510300.SH", "etf", "000300.SH"),
               ("510500.SH", "etf", "000905.SH"),
               ("159845.SZ", "etf", "000932.SH"),
               ("000905.SH", "index", "000905.SH")]


def make_panel(code: str, level: str, idx_code: str, start: str, end: str):
    if level == "index":
        bundle, _ = load_data_index(code, start, end, ROOT)
    elif code == "510300.SH":
        from intramom.config import StrategyConfig
        bundle = load_data(StrategyConfig(symbol=code, start=start, end=end), ROOT)
    elif code == "510500.SH":
        bundle, _ = load_data_ext(code, start, end, ROOT)
    else:
        bundle, _ = load_data_etf(code, start, end, ROOT)
    if level == "etf":
        bundle.index_daily = load_index_daily_df(idx_code, ROOT)  # 真实指数日线（σ 源/底仓基准）
    return PanelData.build(bundle)


def stop_rate_of(panel: PanelData) -> float:
    """stop_rate = 2024-12-31 前指数平均日内收益率绝对值（|close/open−1| 的均值）。"""
    d = panel._idx_daily.copy()
    d = d[d["trade_date"] < STOPRATE_END]
    return float((d["close"] / d["open"] - 1.0).abs().mean())


def yearly_returns(daily: pd.DataFrame) -> pd.Series:
    d = daily.set_index("date")["ret"]
    return d.groupby(d.index.year).apply(lambda x: (1 + x).prod() - 1)


def overlay_first_signal(res, panel: PanelData) -> pd.Series:
    """50% 底仓增强：日收益 = 0.5×B&H + 0.5×(当日第一趟盈亏/策略本金)。"""
    rt = round_trips(res.trades)
    first = rt.sort_values(["date", "entry_time"]).groupby("date").first()
    d = res.daily.set_index("date").copy()
    strat = (first["pnl"] / d["aum_start"]).reindex(d.index).fillna(0.0)
    bh = pd.Series(panel.ret_v, index=panel.days).reindex(d.index).fillna(0.0)
    return 0.5 * bh + 0.5 * strat


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_base_config(ROOT).with_overrides(
        grid="custom", grid_points=("10:29", "11:29", "13:59"), stop_freq="1min",
        entry_mode="band_state", vol_mult=1.0, lookback=14, vol_source="index",
        slippage_per_share=0.0, end=SAMPLE_END)

    chain_rows, overlay_rows = [], []
    for code, level, idx_code in INSTRUMENTS:
        print(f"[wb] {code}({level}) 构建面板 ...", flush=True)
        panel = make_panel(code, level, idx_code, base.start, base.end)
        sr = stop_rate_of(panel)
        print(f"[wb] {code} stop_rate={sr:.4%}", flush=True)
        steps = [
            ("S0 原始策略", {}),
            ("S1 +5minVWAP", {"fill_model": "vwap5"}),
            ("S2 +放松平仓", {"fill_model": "vwap5", "stop_mode": "band_vwap_loose"}),
            ("S3 +收窄m=0.7", {"fill_model": "vwap5", "stop_mode": "band_vwap_loose",
                             "exit_mult": 0.7}),
            ("S4 +阶梯止盈(延迟平仓+止盈)",
             {"fill_model": "vwap5", "stop_mode": "band_vwap_loose",
              "exit_mult": 0.7, "ladder_rate": sr}),
        ]
        curves = {}
        s4_res = None
        for sname, ov in steps:
            cfg = base.with_overrides(symbol=code, **ov)
            res = run_backtest(panel, cfg)
            r = res.daily["ret"].to_numpy()
            st = perf_stats(r, INIT * np.cumprod(1 + r), init=INIT)
            rt = round_trips(res.trades)
            chain_rows.append({"code": code, "level": level, "step": sname,
                               "TotalReturn": st["TotalReturn"], "IRR": st["IRR"],
                               "Sharpe": st["Sharpe"], "MDD": st["MDD"],
                               "Calmar": st["IRR"] / abs(st["MDD"]) if st["MDD"] else np.nan,
                               "win_rate_day": st["HitRatio"], "n_trades": len(rt),
                               "exec_rows": len(res.trades)})
            curves[sname] = res.daily.set_index("date")["ret"]
            if sname.startswith("S4"):
                s4_res = res
        # 年度收益表（S4）
        yr = yearly_returns(s4_res.daily)
        yr.to_csv(OUT / f"yearly_{code.split('.')[0]}.csv")
        # 底仓增强
        ov_ret = overlay_first_signal(s4_res, panel)
        bh = pd.Series(panel.ret_v, index=panel.days).reindex(ov_ret.index).fillna(0.0)
        st_o = perf_stats(ov_ret.to_numpy(), INIT * np.cumprod(1 + ov_ret.to_numpy()), init=INIT)
        st_b = perf_stats(0.5 * bh.to_numpy(), INIT * np.cumprod(1 + 0.5 * bh.to_numpy()), init=INIT)
        overlay_rows.append({"code": code, "增强策略IRR": st_o["IRR"], "50%BH_IRR": st_b["IRR"],
                             "超额年化": st_o["IRR"] - st_b["IRR"],
                             "增强Sharpe": st_o["Sharpe"], "50%BH_Sharpe": st_b["Sharpe"],
                             "增强MDD": st_o["MDD"], "50%BH_MDD": st_b["MDD"]})
        # 净值图
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for sname, r in curves.items():
            ax.plot(r.index, np.cumprod(1 + r.to_numpy()), lw=1.1, label=sname)
        bh_c = np.cumprod(1 + bh.to_numpy())
        ax.plot(ov_ret.index, bh_c, lw=1.0, color="gray", ls="--", label="B&H")
        ax.plot(ov_ret.index, np.cumprod(1 + ov_ret.to_numpy()), lw=1.2, color="black",
                label="S4+底仓增强(50%)")
        ax.set_title(f"{code} 改进链净值（研报口径复现，{panel.days[0].date()}~{panel.days[-1].date()}）")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"chain_{code.split('.')[0]}.png")
        plt.close(fig)
        print(f"[wb] {code} 完成", flush=True)

    chain = pd.DataFrame(chain_rows)
    chain.to_csv(OUT / "chain_summary.csv", index=False)
    overlay = pd.DataFrame(overlay_rows)
    overlay.to_csv(OUT / "overlay_summary.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n===== 改进链对照（2017-01 ~ 2025-10-10，研报口径） =====")
    print(chain.pivot_table(index="step", columns="code", values="Sharpe").to_string())
    print("\n===== 底仓增强（S4 改进策略，每日仅第一信号） =====")
    print(overlay.to_string(index=False))
    print(f"\n[wb] 产物 -> {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
