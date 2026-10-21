#!/usr/bin/env python
"""旗舰策略（h3 网格 + A-L-1：10:29/11:29/13:59 入场判断，次 bar 开盘成交，每分钟止损）
对 ETF 名单批量回测。

名单：data/A股大盘行业ETF名单.csv（板块,代码,ETF名称）
- 510300.SH 用前序冻结快照；510500.SH 用 processed_ext；其余走 MySQL（etf_loader，带缓存）
- 输出：data/results/etf_scan/<code>/（OU 风格 13 文件 + scan_qa.json）+ summary.csv
- 数据加载串行（MySQL 友好），回测与落盘多线程

用法：
    python scripts/run_etf_scan.py                 # 全名单
    python scripts/run_etf_scan.py --only 512880.SH 159928.SZ
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd  # noqa: E402

from intramom.config import StrategyConfig, load_base_config  # noqa: E402
from intramom.data_loader import load_data  # noqa: E402
from intramom.etf_loader import load_data_etf  # noqa: E402
from intramom.ext_loader import load_data_ext  # noqa: E402
from intramom.panel import PanelData  # noqa: E402
from intramom.runner import run_one  # noqa: E402

ETF_LIST = ROOT / "data" / "A股大盘行业ETF名单.csv"
OUT_ROOT = ROOT / "data" / "results" / "etf_scan"

# 旗舰策略参数（h3 A-L-1，全谱最优格）
FLAGSHIP = {
    "grid": "custom", "grid_points": ("10:29", "11:29", "13:59"),
    "stop_mode": "opp_band", "sizing": "fixed", "stop_freq": "1min",
    "vol_mult": 2.0, "lookback": 20,
}


def make_bundle(symbol: str, start: str, end: str):
    """按标的走对应数据通道，返回 (bundle, qa)。"""
    if symbol == "510300.SH":
        cfg = StrategyConfig(symbol=symbol, start=start, end=end)
        bundle = load_data(cfg, ROOT)
        return bundle, {"symbol": symbol, "channel": "frozen RUN-001"}
    if symbol in ("510500.SH", "159915.SZ"):
        bundle, qa = load_data_ext(symbol, start, end, ROOT)
        return bundle, qa
    return load_data_etf(symbol, start, end, ROOT)


def work(symbol: str, sector: str, name: str, cfg: StrategyConfig,
         bundle=None, qa=None) -> dict:
    if bundle is None:
        bundle, qa = make_bundle(symbol, cfg.start, cfg.end)
    panel = PanelData.build(bundle)
    panel._data_dir = str(bundle.data_dir)
    cfg_s = cfg.with_overrides(symbol=symbol, run_name=symbol.split(".")[0])
    out = run_one(cfg_s, panel=panel, project_root=ROOT,
                  out_dir=OUT_ROOT / symbol.split(".")[0], with_anchors=False)
    (Path(out["out_dir"]) / "scan_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"sector": sector, "code": symbol, "name": name,
            "first_day": qa.get("first_day", ""), "n_days": qa.get("n_days", panel.n_days),
            "zero_vol_share": qa.get("zero_vol_share", ""),
            **out["summary"]}


def main() -> int:
    ap = argparse.ArgumentParser(description="旗舰策略 ETF 全名单回测")
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定代码")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    etfs = pd.read_csv(ETF_LIST)
    if args.only:
        etfs = etfs[etfs["代码"].isin(args.only)]
    print(f"[etf_scan] {len(etfs)} 只 ETF，旗舰参数: {FLAGSHIP}", flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    cfg = load_base_config(ROOT).with_overrides(**FLAGSHIP)

    # ---- 串行加载（MySQL 友好），缓存后幂等 ----
    prepared = {}
    for r in etfs.itertuples():
        sym = r.代码
        print(f"[etf_scan] 加载 {sym}({r.ETF名称}) ...", flush=True)
        try:
            prepared[sym] = make_bundle(sym, cfg.start, cfg.end)
            qa = prepared[sym][1]
            print(f"           days={qa.get('n_days', '?')} "
                  f"dropped={len(qa.get('dropped_days_missing_price_labels', []))} "
                  f"close_maxdiff={qa.get('close_maxdiff_vs_daily', 0):.4f} "
                  f"zero_vol={qa.get('zero_vol_share', 0):.1%}"
                  if isinstance(qa.get('zero_vol_share'), float) else "", flush=True)
        except Exception as e:
            print(f"           !! 加载失败: {type(e).__name__}: {e}", flush=True)
            prepared[sym] = e

    # ---- 多线程回测 + 落盘 ----
    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for r in etfs.itertuples():
            sym = r.代码
            if isinstance(prepared.get(sym), Exception):
                failed.append({"code": sym, "stage": "load",
                               "error": str(prepared[sym])[:200]})
                continue
            bundle, qa = prepared[sym]
            futs[ex.submit(work, sym, r.板块, r.ETF名称, cfg, bundle, qa)] = sym
        for fut, sym in futs.items():
            try:
                row = fut.result()
                rows.append(row)
                print(f"[etf_scan] {sym} 完成: TR={row['TotalReturn']:+.2%} "
                      f"Sharpe={row['Sharpe']:+.3f} MDD={row['MDD']:.2%}", flush=True)
            except Exception as e:
                failed.append({"code": sym, "stage": "backtest",
                               "error": f"{type(e).__name__}: {e}"[:300]})
                print(f"[etf_scan] {sym} 回测失败: {e}", flush=True)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("Sharpe", ascending=False)
        df.to_csv(OUT_ROOT / "summary.csv", index=False)
    if failed:
        pd.DataFrame(failed).to_csv(OUT_ROOT / "failed.csv", index=False)

    cols = ["sector", "code", "name", "first_day", "n_days", "TotalReturn",
            "IRR", "Sharpe", "MDD", "HitRatio", "TradedDays"]
    cols = [c for c in cols if c in df.columns]
    print("\n===== 旗舰策略 ETF 扫描（按 Sharpe 排序） =====")
    print(df[cols].to_string(index=False,
                             float_format=lambda x: f"{x:,.3f}"))
    if failed:
        print("\n失败:", failed)
    print(f"\n[etf_scan] 结果目录: {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
