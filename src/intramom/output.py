"""OU_GRID 风格结果落盘：每次运行一个目录，配置/参数/指标/流水/决策/净值/图表/报告齐全。

目录内容（对齐 OU_GRID_STRATEGY_v1.1/data/results 规范）：
config_snapshot.json / params.json / metrics.json / meta.json /
equity_daily.csv / equity_drawdown.png / fills.csv / round_trips.csv /
strategy_decisions.csv / account_snapshots.csv / position_snapshots.csv /
backtest_summary.csv / report.md
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .engine import BacktestResult
from .metrics import perf_stats, round_trips, summary_row, trade_level_stats, ols_stats

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

_PLOT_LOCK = threading.Lock()  # matplotlib 全局状态非线程安全，遍历并发时串行化绘图


def _dump_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def plot_equity_drawdown(daily: pd.DataFrame, bh_nav: np.ndarray, title: str, path: Path) -> None:
    """净值 + 回撤双面板图（策略 vs 标的 B&H）。"""
    with _PLOT_LOCK:
        nav = daily["nav"].to_numpy()
        dd = nav / np.maximum.accumulate(nav) - 1.0
        bh = bh_nav / bh_nav[0]
        bh_dd = bh / np.maximum.accumulate(bh) - 1.0
        x = pd.to_datetime(daily["date"])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})
        ax1.plot(x, nav, lw=1.2, label="strategy")
        ax1.plot(x, bh, lw=1.0, alpha=0.7, label="510300 B&H")
        ax1.set_ylabel("NAV")
        ax1.set_title(title)
        ax1.legend(loc="best")
        ax1.grid(alpha=0.3)
        ax2.fill_between(x, dd, 0, alpha=0.5, label="strategy DD")
        ax2.plot(x, bh_dd, lw=0.8, alpha=0.7, label="B&H DD")
        ax2.set_ylabel("Drawdown")
        ax2.legend(loc="best")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def render_report(res: BacktestResult, summary: dict, tstats: dict, ab: dict,
                  data_info: dict, anchor_rows: list[dict] | None) -> str:
    cfg = res.config
    L = []
    L.append(f"# 日内动量基础策略回测报告 — {cfg.name}")
    L.append("")
    L.append("论文：*Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF "
             "(SPY)*（Zarattini, Aziz, Barbon, SSRN 4824172）的 A 股复现。")
    L.append("")
    L.append("| 回测口径 | 值 |")
    L.append("|---|---|")
    L.append(f"| 标的 | {cfg.symbol}（沪深300ETF 华泰柏瑞） |")
    L.append(f"| 样本区间 | {data_info['first_day']} 至 {data_info['last_day']}（{data_info['n_days']} 交易日） |")
    L.append(f"| 决策网格 | {cfg.grid}（{', '.join(cfg.grid_taus)}） |")
    L.append(f"| 信号 | 噪声带突破：L={cfg.lookback}，VM={cfg.vol_mult:g} |")
    L.append(f"| 止损模式 | {cfg.stop_mode} |")
    L.append(f"| 仓位模式 | {cfg.sizing}"
             + (f"（σ_target={cfg.sigma_target:g}，max_lev={cfg.max_leverage:g}，vol_lookback={cfg.vol_lookback}） |"
                if cfg.sizing == "vol_target" else " |"))
    L.append(f"| 成本 | 佣金 {cfg.commission_rate * 1e4:g}bp/边 + 滑点 {cfg.slippage_per_share:g} 元/份/边 |")
    L.append(f"| 整手 | {cfg.lot_size} 份 |")
    L.append(f"| 初始资金 | {cfg.initial_capital:,.0f} |")
    L.append(f"| 成交假设 | 信号 = τ bar 收盘 → 成交 = 下一分钟 bar 开盘；15:00 收盘强制平仓 |")
    L.append(f"| 数据 | `{data_info['data_dir']}`（前序 RUN-001 冻结快照） |")
    L.append("")
    L.append("| 绩效 | 值 |")
    L.append("|---|---:|")
    L.append(f"| 总收益 | {summary['TotalReturn']:.2%} |")
    L.append(f"| 年化收益(IRR) | {summary['IRR']:.2%} |")
    L.append(f"| 年化波动率 | {summary['Vol']:.2%} |")
    L.append(f"| Sharpe | {summary['Sharpe']:.4f} |")
    L.append(f"| 最大回撤 | {summary['MDD']:.2%} |")
    L.append(f"| 日胜率(全样本/交易日) | {summary['HitRatio']:.2%} / {summary['HitRatioTraded']:.2%} |")
    L.append(f"| 偏度 | {summary['Skew']:.4f} |")
    L.append(f"| 最差/最好日 | {summary['WorstDay']:.2%} / {summary['BestDay']:.2%} |")
    L.append(f"| 平均日收益(bps) | {summary['AvgDailyBps']:.2f} |")
    L.append(f"| 期末 AUM | {res.end_aum:,.2f} |")
    L.append("")
    L.append("| 交易统计 | 值 |")
    L.append("|---|---:|")
    L.append(f"| 整趟交易数 | {tstats['Trades']} |")
    L.append(f"| 日均交易 | {tstats['AvgTradePerDay']:.2f} |")
    L.append(f"| 整趟胜率 | {tstats['HitRatio']:.2%} |")
    L.append(f"| 平均每份盈亏 | {tstats['AvgPnLPerShare']:.5f} |")
    L.append(f"| 最大单笔亏损/盈利 | {tstats['MaxLossTradePct']:.2f}%（{tstats['MaxLossDate']}）/ "
             f"{tstats['MaxGainTradePct']:.2f}%（{tstats['MaxGainDate']}） |")
    L.append(f"| 多头占比 | {tstats['LongShare']:.2%} |")
    L.append("")
    L.append("| 回归（策略日收益 ~ 标的日收益） | 值 |")
    L.append("|---|---:|")
    L.append(f"| α(年化) | {ab['alpha_annual']:.2%}（t={ab['t_alpha']:.2f}） |")
    L.append(f"| β | {ab['beta']:.4f}（t={ab['t_beta']:.2f}） |")
    L.append(f"| R² | {ab['r2']:.4f} |")
    L.append("")
    if anchor_rows:
        L.append("## 与前序 RUN-001 锚点勾稽")
        L.append("")
        L.append("| 锚点 | 期望 | 实际 | 判定 |")
        L.append("|---|---:|---:|---|")
        for r in anchor_rows:
            L.append(f"| {r['name']} | {r['expected']} | {r['actual']} | {r['status']} |")
        L.append("")
    L.append("## 文件清单")
    L.append("")
    L.append("`config_snapshot.json` 完整配置；`params.json` 参数；`metrics.json` 指标；"
             "`fills.csv` 每笔成交；`round_trips.csv` 整趟；`strategy_decisions.csv` 决策明细；"
             "`account_snapshots.csv` / `position_snapshots.csv` 快照；"
             "`equity_daily.csv` / `equity_drawdown.png` 净值与回撤；`backtest_summary.csv` 单行汇总。")
    L.append("")
    return "\n".join(L)


def write_run(res: BacktestResult, panel, out_dir: str | Path,
              anchor_rows: list[dict] | None = None) -> dict:
    """把一次回测结果按 OU 风格落盘，返回写入的文件路径字典。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = res.config
    dl = res.daily.copy()
    dl["nav"] = dl["aum_end"] / cfg.initial_capital
    dl["bh_nav"] = panel.bh_nav

    summary = summary_row(res)
    rt = round_trips(res.trades)
    tstats = trade_level_stats(rt, len(dl))
    ab = ols_stats(dl["ret"].to_numpy(), panel.ret_v)
    data_info = {
        "data_dir": str(panel.__dict__.get("_data_dir", "")),
        "n_days": panel.n_days,
        "first_day": str(panel.days[0].date()),
        "last_day": str(panel.days[-1].date()),
    }

    written = {}
    # ---- 配置与参数 ----
    cfg.to_json(out / "config_snapshot.json")
    _dump_json({
        "signal": {"lookback": cfg.lookback, "vol_mult": cfg.vol_mult,
                   "grid": cfg.grid, "grid_taus": cfg.grid_taus},
        "variant": {"stop_mode": cfg.stop_mode, "sizing": cfg.sizing,
                    "sigma_target": cfg.sigma_target, "max_leverage": cfg.max_leverage,
                    "vol_lookback": cfg.vol_lookback, "stop_freq": cfg.stop_freq},
        "costs": {"commission_rate": cfg.commission_rate,
                  "slippage_per_share": cfg.slippage_per_share, "lot_size": cfg.lot_size},
        "account": {"initial_capital": cfg.initial_capital},
        "sample": {"start": cfg.start, "end": cfg.end, "symbol": cfg.symbol},
    }, out / "params.json")
    # ---- 指标 ----
    metrics = {**{k: v for k, v in summary.items() if k not in ("Run", "Variant", "Grid")},
               "end_aum": res.end_aum,
               "exec_rows": int(len(res.trades)),
               **{f"trade_{k}": v for k, v in tstats.items()},
               "alpha_annual": ab["alpha_annual"], "beta": ab["beta"],
               "t_alpha": ab["t_alpha"], "t_beta": ab["t_beta"], "r2": ab["r2"]}
    _dump_json(metrics, out / "metrics.json")
    _dump_json({"run_name": cfg.name, "created_at": datetime.now(timezone.utc).isoformat(),
                "variant": cfg.variant_tag, "grid": cfg.grid,
                "sample": {"start": data_info["first_day"], "end": data_info["last_day"],
                           "n_days": panel.n_days},
                "data_dir": data_info["data_dir"],
                "exec_rows": int(len(res.trades)), "end_aum": res.end_aum},
               out / "meta.json")
    # ---- 表格 ----
    dl.to_csv(out / "equity_daily.csv", index=False)
    res.trades.to_csv(out / "fills.csv", index=False)
    rt.to_csv(out / "round_trips.csv", index=False)
    res.decisions.to_csv(out / "strategy_decisions.csv", index=False)
    acc = dl[["date", "aum_start", "aum_end", "ret", "lev", "n_exec", "traded"]].copy()
    acc["pnl"] = acc["aum_end"] - acc["aum_start"]
    acc.to_csv(out / "account_snapshots.csv", index=False)
    res.decisions[["date", "tau", "pos_before", "pos_after"]].to_csv(
        out / "position_snapshots.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "backtest_summary.csv", index=False)
    # ---- 图与报告 ----
    plot_equity_drawdown(dl, panel.bh_nav, f"{cfg.name} NAV & Drawdown", out / "equity_drawdown.png")
    (out / "report.md").write_text(
        render_report(res, summary, tstats, ab, data_info, anchor_rows), encoding="utf-8")

    for p in out.iterdir():
        written[p.name] = str(p)
    return {"out_dir": str(out), "files": written, "summary": summary,
            "trade_stats": tstats, "alpha_beta": ab}
