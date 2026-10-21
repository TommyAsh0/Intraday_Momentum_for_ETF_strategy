"""策略配置模型：全部参数一处定义，支持 JSON 序列化与参数遍历展开。

改参/加参指南：
- 改参：直接在 config/base.json 改值，或代码中 dataclasses.replace(cfg, vol_mult=1.5)。
- 加参：在 StrategyConfig 增加带默认值的字段即可；to_dict/from_dict/run_name 自动兼容
  （新参数如需进入 run_name，在 _name_parts() 中追加）。
- 遍历：StrategyConfig.expand(base, vol_mult=[...], lookback=[...]) 笛卡尔展开。
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------- 决策网格
# 决策时点 τ（信号价 = 标签 τ bar 收盘；13:00 用 11:30 bar，成交价 = 下一分钟 bar 开盘）


def make_grid(step_min: int, last_pm: str) -> list[str]:
    """生成决策网格：10:00 起上午每 step_min 一档至 11:00，13:00 起下午至 last_pm。"""
    am, pm = [], []
    t = 10 * 60
    while t <= 11 * 60:
        am.append(f"{t // 60:02d}:{t % 60:02d}")
        t += step_min
    t = 13 * 60
    last = int(last_pm[:2]) * 60 + int(last_pm[3:])
    while t <= last:
        pm.append(f"{t // 60:02d}:{t % 60:02d}")
        t += step_min
    return am + pm


GRIDS: dict[str, list[str]] = {
    "g30": make_grid(30, "14:30"),  # 论文主线：7 时点
    "g15": make_grid(15, "14:45"),  # 14 时点
    "g05": make_grid(5, "14:55"),   # 42 时点
}

STOP_MODES = ("opp_band", "band_vwap", "vwap_only", "band_vwap_loose")
SIZING_MODES = ("fixed", "vol_target")

# 变体标签（论文口径）：A=反向带止损+固定仓位；B=当前带+VWAP 止损+固定仓位；C=B+波动目标动态仓位
# D（论文外组合）=A 的反向带止损+波动目标动态仓位
_VARIANT_TAGS = {
    ("opp_band", "fixed"): "A",
    ("band_vwap", "fixed"): "B",
    ("band_vwap", "vol_target"): "C",
    ("opp_band", "vol_target"): "D",
    ("vwap_only", "fixed"): "VW",
    ("vwap_only", "vol_target"): "VWD",
}


@dataclass(frozen=True)
class StrategyConfig:
    """基础策略完整参数集（默认 = 论文基础口径，510300.SH，2017-2025）。

    参数分组：
    - 数据：symbol / data_dir / start / end
    - 信号：lookback(σ 窗口) / vol_mult(噪声带乘数 VM) / grid / grid_points(自定义网格)
    - 变体：stop_mode / sizing / sigma_target / max_leverage / vol_lookback(日频波动窗口)
    - 成本：commission_rate / slippage_per_share / lot_size
    - 账户：initial_capital
    - 输出：results_root / run_name(空 = 自动生成)
    """

    # ---- 数据 ----
    symbol: str = "510300.SH"
    data_dir: str = "../Intraday_Momentum_for_ETF/data/processed"
    start: str = "2017-01-01"
    end: str = "2025-12-31"
    # ---- 信号 ----
    lookback: int = 14
    vol_mult: float = 1.0
    grid: str = "g30"
    grid_points: tuple[str, ...] | None = None  # grid="custom" 时生效
    # ---- 变体 ----
    stop_mode: str = "opp_band"
    sizing: str = "fixed"
    sigma_target: float = 0.02
    max_leverage: float = 4.0
    vol_lookback: int = 14
    stop_freq: str = "grid"   # "grid"=止损在决策时点检查（论文口径）；"1min"=每分钟检查（L-1 变体）
    vol_source: str = "etf"   # σ_daily 数据源："etf"=标的自身日收益（论文口径）；"index"=对标指数
    entry_mode: str = "new_cross"  # "new_cross"=仅新跨越入场(论文 armed 机制)；
                                   # "band_state"=决策时点价格在带外即入场(券商研报口径)
    fill_model: str = "open"  # "open"=次分钟开盘价成交(市价)；
                              # "passive_chase"=限价挂单（买=open−1tick/卖=+1tick），
                              # 当根 bar 触及限价则成交于限价（省 1 tick），否则以该 bar 收盘追价成交
                              # "vwap5"=信号后 5 分钟 VWAP 成交（研报 2.0 可执行口径）；
                              #         EOD 平仓=最后 5 分钟(14:56..15:00) VWAP
    exit_mult: float = 1.0    # 退出边界收窄系数 m（0<m≤1；1=与入场带相同）；仅收窄止损带，入场带不变
    ladder_rate: float = 0.0  # 阶梯止盈阈值（日内浮动收益率>该值时止盈 ladder_frac 仓位；0=关闭）
    ladder_frac: float = 0.5  # 阶梯止盈平仓比例
    # ---- 成本 ----
    commission_rate: float = 0.0001   # 万1/边
    slippage_per_share: float = 0.001  # 元/份/边（1 tick）
    lot_size: int = 100
    # ---- 账户 ----
    initial_capital: float = 1_000_000.0
    # ---- 输出 ----
    results_root: str = "data/results"
    run_name: str = ""

    # ------------------------------------------------------------ 校验
    def __post_init__(self):
        if self.stop_mode not in STOP_MODES:
            raise ValueError(f"stop_mode 必须为 {STOP_MODES}，得到 {self.stop_mode!r}")
        if self.sizing not in SIZING_MODES:
            raise ValueError(f"sizing 必须为 {SIZING_MODES}，得到 {self.sizing!r}")
        if self.stop_freq not in ("grid", "1min"):
            raise ValueError(f"stop_freq 必须为 'grid' 或 '1min'，得到 {self.stop_freq!r}")
        if self.vol_source not in ("etf", "index"):
            raise ValueError(f"vol_source 必须为 'etf' 或 'index'，得到 {self.vol_source!r}")
        if self.entry_mode not in ("new_cross", "band_state"):
            raise ValueError(f"entry_mode 必须为 'new_cross' 或 'band_state'，得到 {self.entry_mode!r}")
        if self.fill_model not in ("open", "passive_chase", "vwap5"):
            raise ValueError(f"fill_model 必须为 'open'/'passive_chase'/'vwap5'，得到 {self.fill_model!r}")
        if not (0 < self.exit_mult <= 1.0):
            raise ValueError("exit_mult 须在 (0,1]")
        if self.ladder_rate < 0 or not (0.0 < self.ladder_frac <= 1.0):
            raise ValueError("ladder_rate >= 0 且 ladder_frac ∈ (0,1]")
        if self.lookback < 1 or self.vol_lookback < 2:
            raise ValueError("lookback >= 1 且 vol_lookback >= 2")
        if self.grid != "custom" and self.grid not in GRIDS:
            raise ValueError(f"未知 grid {self.grid!r}，可选 {list(GRIDS)} 或 'custom'")
        if self.grid == "custom" and not self.grid_points:
            raise ValueError("grid='custom' 时必须提供 grid_points")
        if self.lot_size < 1 or self.initial_capital <= 0:
            raise ValueError("lot_size >= 1 且 initial_capital > 0")

    # ------------------------------------------------------------ 派生
    @property
    def grid_taus(self) -> list[str]:
        return list(self.grid_points) if self.grid == "custom" else list(GRIDS[self.grid])

    @property
    def variant_tag(self) -> str:
        base = _VARIANT_TAGS.get((self.stop_mode, self.sizing),
                                 f"{self.stop_mode}+{self.sizing}")
        return f"{base}-L-1" if self.stop_freq == "1min" else base

    def auto_name(self) -> str:
        """由参数确定性生成运行名（相同参数 → 相同目录名）。"""
        parts = [self.variant_tag, self.grid,
                 f"VM{self.vol_mult:g}", f"L{self.lookback}"]
        if self.sizing == "vol_target":
            parts.append(f"st{self.sigma_target:g}x{self.max_leverage:g}")
            if self.vol_source == "index":
                parts.append("vi")
        if self.entry_mode == "band_state":
            parts.append("bs")
        if self.fill_model == "passive_chase":
            parts.append("pc")
        elif self.fill_model == "vwap5":
            parts.append("v5")
        if self.exit_mult != 1.0:
            parts.append(f"xm{self.exit_mult:g}")
        if self.ladder_rate > 0:
            parts.append(f"ld{self.ladder_rate:g}")
        parts.append(f"c{self.commission_rate * 1e4:g}bp")
        parts.append(f"s{self.slippage_per_share / 0.001:g}t")
        return "_".join(parts)

    @property
    def name(self) -> str:
        return self.run_name or self.auto_name()

    # ------------------------------------------------------------ 序列化
    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        if d["grid_points"] is not None:
            d["grid_points"] = list(d["grid_points"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategyConfig":
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        if d.get("grid_points") is not None:
            d["grid_points"] = tuple(d["grid_points"])
        return cls(**d)

    @classmethod
    def from_json(cls, path: str | Path) -> "StrategyConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    # ------------------------------------------------------------ 遍历
    def with_overrides(self, **kw: Any) -> "StrategyConfig":
        """改参入口：返回替换参数后的新配置（原配置不变）。"""
        if "run_name" not in kw:
            kw["run_name"] = ""  # 改参后重新自动命名，除非显式指定
        return replace(self, **kw)

    @classmethod
    def expand(cls, base: "StrategyConfig", **param_values: Iterable[Any]) -> list["StrategyConfig"]:
        """参数遍历：对给定字段取笛卡尔积，返回全部组合（顺序确定）。"""
        keys = list(param_values.keys())
        vals = [list(param_values[k]) for k in keys]
        combos = []
        for prod in itertools.product(*vals):
            combos.append(base.with_overrides(**dict(zip(keys, prod))))
        return combos


def load_base_config(project_root: str | Path, overrides: dict[str, Any] | None = None) -> StrategyConfig:
    """读取 config/base.json 并可覆盖参数。"""
    cfg = StrategyConfig.from_json(Path(project_root) / "config" / "base.json")
    return cfg.with_overrides(**overrides) if overrides else cfg
