#!/usr/bin/env python
"""扩展标的（510500.SH / 159915.SZ）基础策略回测（默认变体 A g15），OU 风格落盘。

用法：
    python scripts/run_ext_base.py                      # 两个扩展标的都跑（变体 A g15）
    python scripts/run_ext_base.py --symbols 510500.SH --variant B --grid g30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd  # noqa: E402

from intramom.config import StrategyConfig, load_base_config  # noqa: E402
from intramom.ext_loader import EXT_SYMBOLS, load_data_ext  # noqa: E402
from intramom.panel import PanelData  # noqa: E402
from intramom.runner import VARIANT_PARAMS, run_one  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="扩展标的 base 策略回测")
    ap.add_argument("--symbols", nargs="+", default=list(EXT_SYMBOLS))
    ap.add_argument("--variant", default="A", choices=["A", "B", "C"])
    ap.add_argument("--grid", default="g15", choices=["g30", "g15", "g05"])
    args = ap.parse_args()

    base = load_base_config(ROOT).with_overrides(grid=args.grid,
                                                 **VARIANT_PARAMS[args.variant])
    summaries = []
    for sym in args.symbols:
        code = sym.split(".")[0]
        print(f"[ext] {sym} 加载数据 ...", flush=True)
        bundle, qa = load_data_ext(sym, base.start, base.end, ROOT)
        print(f"[ext] {sym} QA: days={qa['n_days']} dropped={len(qa['dropped_days_missing_price_labels'])} "
              f"close_maxdiff={qa['close_maxdiff_vs_daily']:.4f} zero_vol={qa['zero_vol_bars']}")
        panel = PanelData.build(bundle)
        panel._data_dir = str(bundle.data_dir)
        cfg = base.with_overrides(symbol=sym,
                                  run_name=f"ext_{code}_{args.variant}_{args.grid}")
        out = run_one(cfg, panel=panel, project_root=ROOT, with_anchors=False)
        s = out["summary"]
        summaries.append({"symbol": sym, **s})
        print(f"[ext] {sym} {args.variant}/{args.grid}: exec={len(out['result'].trades)} "
              f"end_aum={out['result'].end_aum:,.2f} Sharpe={s['Sharpe']:.4f} "
              f"MDD={s['MDD']:.2%} -> {out['out_dir']}", flush=True)
        (Path(out["out_dir"]) / "ext_qa.json").write_text(
            json.dumps(qa, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    df = pd.DataFrame(summaries)
    cols = ["symbol", "Variant", "Grid", "N", "TradedDays", "TotalReturn", "IRR",
            "Vol", "Sharpe", "MDD"]
    print("\n===== 扩展标的汇总 =====")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
