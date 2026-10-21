"""端到端运行器：数据 → 面板 → 回测 → 指标 → （可选）OU 风格落盘。"""
from __future__ import annotations

from pathlib import Path

from .anchors import check_anchors
from .config import StrategyConfig
from .data_loader import load_data
from .engine import BacktestResult, run_backtest
from .metrics import summary_row
from .output import write_run
from .panel import PanelData

# 变体标签 → (stop_mode, sizing)，对应论文 A/B/C；D=论文外组合（A 止损 + 动态仓位）
VARIANT_PARAMS = {
    "A": {"stop_mode": "opp_band", "sizing": "fixed"},
    "B": {"stop_mode": "band_vwap", "sizing": "fixed"},
    "C": {"stop_mode": "band_vwap", "sizing": "vol_target"},
    "D": {"stop_mode": "opp_band", "sizing": "vol_target"},
}


def make_variant_cfg(base: StrategyConfig, variant: str) -> StrategyConfig:
    """由基础配置生成论文变体配置。"""
    return base.with_overrides(**VARIANT_PARAMS[variant.upper()])


def build_panel(cfg: StrategyConfig, project_root: str | Path) -> PanelData:
    """加载数据并构建面板（每个进程一次，遍历共享）。"""
    bundle = load_data(cfg, project_root)
    panel = PanelData.build(bundle)
    panel._data_dir = str(bundle.data_dir)  # 供 config_snapshot 记录
    return panel


def run_one(cfg: StrategyConfig, panel: PanelData | None = None,
            project_root: str | Path | None = None,
            write: bool = True, out_dir: str | Path | None = None,
            with_anchors: bool = True) -> dict:
    """运行单个参数组合。panel 为空时自行构建（单跑场景）；遍历场景传入共享面板。"""
    if panel is None:
        if project_root is None:
            raise ValueError("panel 为空时必须提供 project_root")
        panel = build_panel(cfg, project_root)
    res = run_backtest(panel, cfg)
    summary = summary_row(res)
    anchor_rows = check_anchors(res, summary) if with_anchors else []
    out = {"result": res, "summary": summary, "anchor_rows": anchor_rows, "out_dir": None}
    if write:
        if out_dir is None:
            if project_root is None:
                raise ValueError("write=True 且 out_dir 为空时必须提供 project_root")
            root = Path(cfg.results_root)
            if not root.is_absolute():
                root = Path(project_root) / root
            out_dir = root / cfg.name
        out.update(write_run(res, panel, out_dir, anchor_rows=anchor_rows or None))
        out["out_dir"] = str(out_dir)
    return out
