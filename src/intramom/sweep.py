"""多线程参数遍历：面板一次构建、只读共享，参数组合并发回测。

用法：
    sweep_df = run_sweep(base_cfg, {"vol_mult": [0.5, 1.0, 1.5], "lookback": [10, 14, 20]},
                         project_root=ROOT, sweep_name="vm_x_lookback")
每个组合写入 data/results/sweeps/<sweep_name>/<run_name>/（OU 风格），
汇总写入 data/results/sweeps/<sweep_name>/summary.csv。

线程安全说明：PanelData 构建后只读（σ/VWAP 缓存键不同则不冲突——遍历时所有
lookback/grid 组合先在主线程预填充缓存，worker 只读）；落盘目录按 run_name 互斥；
matplotlib 绘图在 output 模块内加锁。
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .config import StrategyConfig
from .panel import PanelData
from .runner import build_panel, run_one


def prefill_panel_cache(panel: PanelData, combos: list[StrategyConfig]) -> None:
    """主线程预填充全部缓存键，保证 worker 线程对面板纯只读。"""
    for cfg in combos:
        panel.sigma_panel(cfg.lookback)
        panel.grid_spec(cfg.grid_taus)
        if cfg.stop_mode in ("band_vwap", "vwap_only", "band_vwap_loose"):
            if cfg.stop_freq == "1min":
                panel.vwap_1min_l()
            else:
                panel.vwap_panel(cfg.grid_taus)
        if cfg.fill_model == "vwap5":
            panel.vwap5_l()
        if cfg.sizing == "vol_target":
            panel.sigma_daily(cfg.vol_lookback, cfg.vol_source)


def run_sweep(base_cfg: StrategyConfig, param_grid: dict,
              project_root: str | Path, sweep_name: str,
              n_workers: int | None = None, panel: PanelData | None = None,
              write_runs: bool = True) -> pd.DataFrame:
    """参数遍历入口。

    param_grid: 字段名 → 候选值列表（笛卡尔积），如 {"vol_mult": [1.0, 1.5], "grid": ["g30", "g15"]}。
    返回按组合顺序的汇总 DataFrame（含参数列 + 绩效列）。
    """
    combos = StrategyConfig.expand(base_cfg, **param_grid)
    if len({c.name for c in combos}) != len(combos):
        raise ValueError("存在重名组合（run_name 冲突），请检查遍历参数是否都已纳入 auto_name")
    if panel is None:
        panel = build_panel(base_cfg, project_root)
    prefill_panel_cache(panel, combos)

    sweep_dir = Path(base_cfg.results_root)
    if not sweep_dir.is_absolute():
        sweep_dir = Path(project_root) / sweep_dir
    sweep_dir = sweep_dir / "sweeps" / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    n_workers = n_workers or min(32, (os.cpu_count() or 4))

    def _work(cfg: StrategyConfig) -> dict:
        out = run_one(cfg, panel=panel, project_root=project_root,
                      write=write_runs,
                      out_dir=(sweep_dir / cfg.name) if write_runs else None,
                      with_anchors=False)
        row = {**{k: getattr(cfg, k) for k in param_grid}, **out["summary"]}
        return row

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        rows = list(ex.map(_work, combos))  # map 保持组合顺序

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(sweep_dir / "summary.csv", index=False)
    with open(sweep_dir / "sweep_config.json", "w", encoding="utf-8") as f:
        json.dump({"sweep_name": sweep_name, "base": base_cfg.to_dict(),
                   "param_grid": {k: list(v) for k, v in param_grid.items()},
                   "n_combos": len(combos), "n_workers": n_workers},
                  f, ensure_ascii=False, indent=2, default=str)
    return sweep_df
