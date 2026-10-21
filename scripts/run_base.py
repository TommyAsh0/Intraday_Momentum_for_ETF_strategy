#!/usr/bin/env python
"""复现论文基础策略（510300.SH）：默认跑 A/B/C 三变体（g30），可选锚点验证。

用法：
    python scripts/run_base.py                     # A/B/C 三变体 g30，写入 data/results/base_<tag>_g30/
    python scripts/run_base.py --validate          # 同上 + 对照前序 RUN-001 锚点断言
    python scripts/run_base.py --variant A --grid g15
    python scripts/run_base.py --no-write          # 只回测不落盘
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，统一 UTF-8 输出

import pandas as pd

from intramom import load_base_config, build_panel, make_variant_cfg, run_one
from intramom.anchors import anchors_pass


def main() -> int:
    ap = argparse.ArgumentParser(description="复现论文基础策略（510300.SH）")
    ap.add_argument("--variant", default="all", choices=["A", "B", "C", "all"],
                    help="论文变体：A=反向带止损(基础策略)；B=当前带+VWAP 止损；C=B+动态仓位")
    ap.add_argument("--grid", default="g30", choices=["g30", "g15", "g05"],
                    help="决策网格（论文主线 g30）")
    ap.add_argument("--stop-freq", default="grid", choices=["grid", "1min"],
                    help="止损检查频率：grid=决策时点(论文口径)；1min=每分钟(L-1 变体)")
    ap.add_argument("--config", default=str(ROOT / "config" / "base.json"))
    ap.add_argument("--validate", action="store_true", help="对照前序 RUN-001 锚点断言")
    ap.add_argument("--no-write", action="store_true", help="只回测不落盘")
    args = ap.parse_args()

    from intramom.config import StrategyConfig
    base = StrategyConfig.from_json(args.config)
    base = base.with_overrides(grid=args.grid, stop_freq=args.stop_freq)

    print(f"[run_base] 构建面板（{base.symbol} {base.start}~{base.end}）...")
    panel = build_panel(base, ROOT)
    print(f"[run_base] 面板就绪: {panel.n_days} 交易日 × {len(panel.labels)} 标签")

    variants = ["A", "B", "C"] if args.variant == "all" else [args.variant]
    summaries, all_anchor_rows, failed = [], [], False
    for v in variants:
        cfg0 = make_variant_cfg(base, v)
        cfg = cfg0.with_overrides(run_name=f"base_{cfg0.variant_tag}_{args.grid}")
        out = run_one(cfg, panel=panel, project_root=ROOT, write=not args.no_write)
        s = out["summary"]
        summaries.append(s)
        tag = f"[{v}/{args.grid}] exec={len(out['result'].trades)} " \
              f"end_aum={out['result'].end_aum:,.2f} Sharpe={s['Sharpe']:.4f} MDD={s['MDD']:.2%}"
        if out["out_dir"]:
            tag += f" -> {out['out_dir']}"
        print(tag)
        if args.validate and out["anchor_rows"]:
            all_anchor_rows.extend(out["anchor_rows"])
            if not anchors_pass(out["anchor_rows"]):
                failed = True

    df = pd.DataFrame(summaries)
    cols = ["Variant", "Grid", "N", "TradedDays", "TotalReturn", "IRR", "Vol",
            "Sharpe", "HitRatio", "MDD", "AvgDailyBps"]
    print("\n===== 汇总 =====")
    print(df[cols].to_string(index=False,
                             float_format=lambda x: f"{x:,.4f}"))

    if args.validate:
        print("\n===== RUN-001 锚点勾稽 =====")
        if not all_anchor_rows:
            print("（当前变体×网格无锚点，仅 g30 有全量锚点）")
        for r in all_anchor_rows:
            print(f"  [{r['status']}] {r['name']}: expected={r['expected']} actual={r['actual']}")
        if failed:
            print("锚点验证: FAIL")
            return 1
        if all_anchor_rows:
            print("锚点验证: 全部 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
