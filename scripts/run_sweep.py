#!/usr/bin/env python
"""参数遍历示例（多线程）：面板一次构建，参数组合并发回测。

默认示例：变体 A，g30，vol_mult × lookback 小网格（演示 + 正确性校验）。
按需修改 PARAM_GRID 或用命令行：

    python scripts/run_sweep.py --sweep-name demo_vm_L
    python scripts/run_sweep.py --grids g15 g05 --sweep-name vm_L_g15_g05
    python scripts/run_sweep.py --variant C --workers 16 --sweep-name c_vm_scan

结果：data/results/sweeps/<sweep_name>/summary.csv + 每组合 OU 风格目录。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，统一 UTF-8 输出

from intramom import load_base_config, run_sweep  # noqa: E402
from intramom.runner import VARIANT_PARAMS  # noqa: E402

# 默认遍历网格（示例：噪声带乘数 × σ 窗口）——后期研究在此扩展/替换
PARAM_GRID = {
    "vol_mult": [0.5, 1.0, 1.5, 2.0],
    "lookback": [10, 14, 20],
}


def main() -> int:
    ap = argparse.ArgumentParser(description="日内动量策略参数遍历（多线程）")
    ap.add_argument("--variant", default="A", choices=["A", "B", "C", "D"],
                    help="A/B/C=论文变体；D=论文外组合(A 反向带止损+动态仓位)")
    ap.add_argument("--stop-freq", default="grid", choices=["grid", "1min"],
                    help="止损检查频率：grid=决策时点(论文口径)；1min=每分钟(L-1 变体)")
    ap.add_argument("--grids", nargs="+", default=["g30"],
                    choices=["g30", "g15", "g05", "custom"],
                    help="决策网格（可多个；custom 时需配合 --grid-points）")
    ap.add_argument("--grid-points", nargs="+", default=None,
                    help="自定义决策时点（如 10:29 11:29 13:59）；仅 grids=custom 时生效，"
                         "成交 = 各时点下一分钟 bar 开盘价")
    ap.add_argument("--sweep-name", default="demo_vm_L")
    ap.add_argument("--workers", type=int, default=None, help="线程数（默认 min(32, cpu)）")
    ap.add_argument("--no-write", action="store_true", help="只汇总不落单组合目录")
    args = ap.parse_args()

    base = load_base_config(ROOT).with_overrides(stop_freq=args.stop_freq,
                                                 **VARIANT_PARAMS[args.variant])
    if args.grids == ["custom"]:
        if not args.grid_points:
            raise SystemExit("grids=custom 时必须提供 --grid-points（如 10:29 11:29 13:59）")
        base = base.with_overrides(grid="custom", grid_points=tuple(args.grid_points))
        param_grid = dict(PARAM_GRID)  # custom 网格唯一，不再并入 grid 维度
    else:
        param_grid = {**PARAM_GRID, "grid": args.grids}

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    print(f"[run_sweep] 变体 {args.variant} grids={args.grids} 组合数={n_combos}")
    df = run_sweep(base, param_grid, project_root=ROOT, sweep_name=args.sweep_name,
                   n_workers=args.workers, write_runs=not args.no_write)
    cols = ["grid", "vol_mult", "lookback", "TotalReturn", "IRR", "Sharpe", "MDD",
            "TradedDays", "HitRatio"]
    cols = [c for c in cols if c in df.columns]
    print(f"\n===== sweep 汇总（{args.sweep_name}，按 Sharpe 排序） =====")
    print(df.sort_values("Sharpe", ascending=False)[cols]
          .to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"\n结果目录: data/results/sweeps/{args.sweep_name}/")

    # 多线程正确性自检：同一组合线程并发结果 == 串行重跑结果（抽第 1 个组合）
    from intramom import build_panel  # noqa: E402
    from intramom.runner import run_one  # noqa: E402
    from intramom.config import StrategyConfig  # noqa: E402
    cfg0 = StrategyConfig.expand(base, **param_grid)[0]
    panel = build_panel(base, ROOT)
    serial = run_one(cfg0.with_overrides(), panel=panel, write=False)["summary"]
    row0 = df.iloc[0]
    same = abs(serial["TotalReturn"] - row0["TotalReturn"]) < 1e-12 \
        and abs(serial["Sharpe"] - row0["Sharpe"]) < 1e-12
    print(f"并发/串行一致性自检: {'PASS' if same else 'FAIL'} "
          f"(TR={row0['TotalReturn']:.6f})")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
