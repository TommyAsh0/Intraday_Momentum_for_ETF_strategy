"""Intraday Momentum for ETF — 论文基础策略复现（510300.SH）。

架构：面板预处理（PanelData，一次构建只读共享）+ 参数化回测引擎 + 多线程参数遍历。
论文：Beat the Market — An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)
（Zarattini, Aziz, Barbon, SSRN 4824172）的 A 股 510300 复现。
"""
from .config import GRIDS, StrategyConfig, load_base_config
from .data_loader import DataBundle, load_data
from .panel import PanelData
from .engine import BacktestResult, run_backtest
from .metrics import (perf_stats, ols_stats, summary_row, round_trips,
                      trade_level_stats)
from .runner import build_panel, make_variant_cfg, run_one
from .sweep import run_sweep

__version__ = "0.1.0"
__all__ = [
    "GRIDS", "StrategyConfig", "load_base_config",
    "DataBundle", "load_data", "PanelData",
    "BacktestResult", "run_backtest",
    "perf_stats", "ols_stats", "summary_row", "round_trips", "trade_level_stats",
    "build_panel", "make_variant_cfg", "run_one", "run_sweep",
]
