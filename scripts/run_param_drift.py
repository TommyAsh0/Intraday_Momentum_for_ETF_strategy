#!/usr/bin/env python
"""参数漂移研究：逐年 / 逐半年 / 逐月重校准（walk-forward）。

策略：
- flagship：h3 A-L-1（旗舰配置），调参空间 VM ∈ {1.0,1.5,2.0,2.5,3.0} × L ∈ {10,14,20,30}
- broker2.0：券商研报 2.0 最终版（band_state + vwap5 + band_vwap_loose + 阶梯止盈），
  调参空间 VM ∈ {1.0,1.5,2.0} × L ∈ {10,14,20} × m(exit_mult) ∈ {0.7,0.85,1.0}

方法：
- 每个参数组合全样本回测一次（日内闭合 → 日收益可切片；份额取整的路径依赖误差 <1% 相对，已登记）
- 重校准日（每年/半年/月首个交易日，自 2018-01 起）：取过去 250 个交易日窗口，
  选窗口内 Sharpe 最优组合，应用到下一重校准日（链式样本外）
- 漂移度量：参数变动频率、|Δ参数|、最优参数分布、窗口内最优 Sharpe vs 实际 OOS Sharpe 差距

产物：data/results/param_drift/{drift_<strat>_<freq>.csv, drift_summary.csv, drift_fig.png}
"""
from __future__ import annotations

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

from intramom import load_base_config, build_panel
from intramom.engine import run_backtest
from intramom.metrics import perf_stats

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

OUT = ROOT / "data" / "results" / "param_drift"
INIT = 1_000_000.0
WINDOW = 250          # 调参回看窗口（交易日）
START_RECAL = pd.Timestamp("2018-01-01")
INDEX_OF = {"510300.SH": "000300.SH", "510500.SH": "000905.SH", "159845.SZ": "000932.SH",
            "515220.SH": None, "159852.SZ": None}

BASE = dict(grid="custom", grid_points=("10:29", "11:29", "13:59"), stop_freq="1min",
            vol_source="index")

STRATEGIES = {
    "flagship": {
        "fixed": dict(stop_mode="opp_band", sizing="fixed", entry_mode="new_cross",
                      slippage_per_share=0.001, vol_mult=2.0, lookback=20),
        "grid": {"vol_mult": [1.0, 1.5, 2.0, 2.5, 3.0], "lookback": [10, 14, 20, 30]},
        "param_cols": ["vol_mult", "lookback"],
    },
    "broker2.0": {
        "fixed": dict(stop_mode="band_vwap_loose", sizing="fixed", entry_mode="band_state",
                      fill_model="vwap5", slippage_per_share=0.0, vol_mult=1.0, lookback=14,
                      exit_mult=0.7),
        "grid": {"vol_mult": [1.0, 1.5, 2.0], "lookback": [10, 14, 20],
                 "exit_mult": [0.7, 0.85, 1.0]},
        "param_cols": ["vol_mult", "lookback", "exit_mult"],
    },
}

FREQS = {"yearly": "Y", "semi": "6M", "monthly": "M"}


def recal_dates(days: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """每年/半年/月首个交易日（自 START_RECAL 起）。"""
    d = days[days >= START_RECAL]
    if freq == "Y":
        key = d.year
    elif freq == "6M":
        key = d.year * 2 + (d.month > 6).astype(int)
    else:
        key = d.year * 12 + d.month
    return pd.DatetimeIndex(d.to_series().groupby(key).min().tolist())


def window_sharpe(ret_mat: np.ndarray, dates: pd.DatetimeIndex, t0: pd.Timestamp) -> np.ndarray:
    """t0 前 250 个交易日窗口内各组合的 Sharpe。"""
    i1 = dates.searchsorted(t0)
    lo = max(0, i1 - WINDOW)
    w = ret_mat[lo:i1]
    if len(w) < 60:
        return np.full(ret_mat.shape[1], np.nan)
    mu = w.mean(axis=0)
    sd = w.std(axis=0, ddof=1)
    return np.where(sd > 0, mu / sd * np.sqrt(252), np.nan)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["510300.SH"],
                    choices=["510300.SH", "510500.SH", "159845.SZ", "515220.SH", "159852.SZ"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_base_config(ROOT).with_overrides(**BASE)
    all_summary = []
    for symbol in args.symbols:
        all_summary.extend(run_symbol(base, symbol))
    summ = pd.DataFrame(all_summary)
    summ.to_csv(OUT / "drift_summary.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print(summ.to_string(index=False))
    print(f"[drift] 产物 -> {OUT}/")
    return 0


def run_symbol(base, symbol: str) -> list:
    from intramom.etf_loader import load_data_etf
    from intramom.ext_loader import load_data_ext
    from intramom.data_loader import load_data
    from intramom.config import StrategyConfig
    from intramom.panel import PanelData
    if symbol == "510300.SH":
        bundle = load_data(StrategyConfig(symbol=symbol, start=base.start, end=base.end), ROOT)
    elif symbol == "510500.SH":
        bundle, _ = load_data_ext(symbol, base.start, base.end, ROOT)
    else:
        bundle, _ = load_data_etf(symbol, base.start, base.end, ROOT)
    panel = PanelData.build(bundle)
    days = panel.days
    print(f"[drift] {symbol} 面板 {len(days)} 日", flush=True)
    d = panel._idx_daily.copy()
    d = d[d["trade_date"] < pd.Timestamp("2025-01-01")]
    sr = float((d["close"] / d["open"] - 1.0).abs().mean())
    STRATEGIES["broker2.0"]["fixed"]["ladder_rate"] = sr
    summary_rows = []
    fig, axes = plt.subplots(len(STRATEGIES), 2, figsize=(14, 4.2 * len(STRATEGIES)))

    for si, (sname, spec) in enumerate(STRATEGIES.items()):
        # ---- 全组合全样本回测一次 ----
        keys = spec["grid"]
        import itertools
        combos = [dict(zip(keys, vals))
                  for vals in itertools.product(*[keys[k] for k in keys])]
        ret_cols, fixed_ret = [], None
        combo_names = []
        for ov in combos:
            cfg = base.with_overrides(symbol=symbol, **{**spec["fixed"], **ov})
            res = run_backtest(panel, cfg)
            ret_cols.append(res.daily["ret"].to_numpy())
            combo_names.append(tuple(ov[k] for k in keys))
            if all(spec["fixed"].get(k) == ov.get(k, spec["fixed"].get(k)) for k in keys):
                fixed_ret = ret_cols[-1]
        R = np.column_stack(ret_cols)  # days × combos
        # 固定参数组合（基准对照）
        if fixed_ret is None:
            cfg = base.with_overrides(symbol=symbol, **spec["fixed"])  # 固定参数基准
            fixed_ret = run_backtest(panel, cfg).daily["ret"].to_numpy()

        for fname, fcode in FREQS.items():
            rdates = recal_dates(days, fcode)
            chosen_idx, chosen_params, rows = [], [], []
            chained = np.full(len(days), np.nan)
            for j, t0 in enumerate(rdates):
                t1 = rdates[j + 1] if j + 1 < len(rdates) else days[-1] + pd.Timedelta(days=1)
                shp = window_sharpe(R, days, t0)
                if np.isnan(shp).all():
                    continue
                bi = int(np.nanargmax(shp))
                i0, i1 = days.searchsorted(t0), days.searchsorted(t1)
                oos = R[i0:i1, bi]
                chosen_idx.append(bi)
                chosen_params.append(combo_names[bi])
                oos_st = perf_stats(oos, INIT * np.cumprod(1 + oos), init=INIT) \
                    if len(oos) > 5 else {"Sharpe": np.nan}
                rows.append({"recal_date": t0,
                             **{k: v for k, v in zip(keys, combo_names[bi])},
                             "window_sharpe": shp[bi], "oos_sharpe": oos_st["Sharpe"],
                             "oos_ret": float((1 + oos).prod() - 1) if len(oos) else np.nan})
                chained[i0:i1] = oos
            dfr = pd.DataFrame(rows)
            dfr.to_csv(OUT / f"drift_{sname}_{fname}_{symbol.split('.')[0]}.csv", index=False)

            # 漂移度量
            ch = np.array([chosen_idx[i] != chosen_idx[i - 1]
                           for i in range(1, len(chosen_idx))])
            change_rate = ch.mean() if len(ch) else np.nan
            pmat = np.array(chosen_params, dtype=float)
            prange = pmat.max(axis=0) - pmat.min(axis=0)
            pstd = pmat.std(axis=0)
            # 链式 OOS 指标
            m = ~np.isnan(chained)
            st_chain = perf_stats(chained[m], INIT * np.cumprod(1 + chained[m]), init=INIT)
            st_fixed = perf_stats(fixed_ret[m], INIT * np.cumprod(1 + fixed_ret[m]), init=INIT)
            gap = (dfr["window_sharpe"] - dfr["oos_sharpe"]).mean()
            summary_rows.append({
                "symbol": symbol, "strategy": sname, "freq": fname, "n_recal": len(dfr),
                "param_change_rate": change_rate,
                **{f"range_{k}": prange[i] for i, k in enumerate(keys)},
                **{f"std_{k}": pstd[i] for i, k in enumerate(keys)},
                "window_sharpe_mean": dfr["window_sharpe"].mean(),
                "oos_sharpe_mean": dfr["oos_sharpe"].mean(),
                "overfit_gap": gap,
                "chain_Sharpe": st_chain["Sharpe"], "chain_TR": st_chain["TotalReturn"],
                "chain_MDD": st_chain["MDD"],
                "fixed_Sharpe": st_fixed["Sharpe"], "fixed_TR": st_fixed["TotalReturn"],
                "fixed_MDD": st_fixed["MDD"]})
            print(f"[drift] {symbol} {sname}/{fname}: 变动率={change_rate:.0%} "
                  f"chainSharpe={st_chain['Sharpe']:.3f} vs fixed={st_fixed['Sharpe']:.3f}",
                  flush=True)

        # 图：左=逐月参数时间线；右=链式净值 vs 固定
        ax = axes[si, 0]
        dfr_m = pd.read_csv(OUT / f"drift_{sname}_monthly_{symbol.split('.')[0]}.csv")
        for i, k in enumerate(keys):
            ax.step(pd.to_datetime(dfr_m["recal_date"]), dfr_m[k], where="post",
                    label=k, lw=1.2)
        ax.set_title(f"{sname} 逐月最优参数时间线")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax = axes[si, 1]
        for fname, fcode in FREQS.items():
            dfr_f = pd.read_csv(OUT / f"drift_{sname}_{fname}_{symbol.split('.')[0]}.csv")
            # 从 per-period 记录重建链式净值略重——此处画窗口 Sharpe vs OOS Sharpe 散点
            ax.scatter(dfr_f["window_sharpe"], dfr_f["oos_sharpe"],
                       label=fname, s=14, alpha=0.7)
        lim = [-2, 3]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        ax.set_xlabel("窗口内最优 Sharpe")
        ax.set_ylabel("实际 OOS Sharpe")
        ax.set_title(f"{sname} 调参窗口 vs 样本外（过拟合差距）")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"参数漂移研究（{symbol}，回看窗口 250 交易日）")
    fig.tight_layout()
    fig.savefig(OUT / f"drift_fig_{symbol.split('.')[0]}.png")
    plt.close(fig)
    return summary_rows


if __name__ == "__main__":
    sys.exit(main())
