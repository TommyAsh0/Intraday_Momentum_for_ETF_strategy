#!/usr/bin/env python
"""抓取并冻结 ETF 期权数据（合约表 + 上市日起日行情），含 QA。多标的。

用法：
    python scripts/build_option_data.py --underlying 300etf            # 单标的（断点续抓）
    python scripts/build_option_data.py --underlying 500etf cyb        # 多标的
    python scripts/build_option_data.py --underlying 500etf --basic-only
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

from intramom.options import UNDERLYINGS, build_option_data  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="ETF 期权数据构建（Tushare → data/options/）")
    ap.add_argument("--underlying", nargs="+", default=["300etf"],
                    choices=list(UNDERLYINGS) + ["all"])
    ap.add_argument("--basic-only", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.25, help="API 间隔秒数")
    ap.add_argument("--end", default="20251231")
    args = ap.parse_args()

    keys = list(UNDERLYINGS) if "all" in args.underlying else args.underlying
    cal = pd.read_csv(ROOT.parent / "Intraday_Momentum_for_ETF" / "data" / "processed"
                      / "trade_calendar.csv", dtype={"cal_date": str})
    cal["ymd"] = cal["cal_date"].str.replace("-", "")  # Tushare 要求 YYYYMMDD
    cal = cal[cal.is_open == 1]

    rc = 0
    for key in keys:
        start = UNDERLYINGS[key]["opt_start"]
        days = cal[(cal.ymd >= start) & (cal.ymd <= args.end)]["ymd"].tolist()
        print(f"[build_option_data] {key}({UNDERLYINGS[key]['keyword']}): "
              f"{len(days)} 交易日 ({days[0]}..{days[-1]})", flush=True)
        qa = build_option_data(ROOT, key, days, sleep_s=args.sleep,
                               basic_only=args.basic_only)
        print(json.dumps(qa, ensure_ascii=False, indent=2, default=str))
        ok = (args.basic_only
              or (qa.get("days_mismatch_alive_vs_rows", 1) == 0
                  and not qa["fetch_stats"]["failed_days"]))
        print(f"QA[{key}]:", "PASS" if ok else "CHECK", flush=True)
        rc |= 0 if ok else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
