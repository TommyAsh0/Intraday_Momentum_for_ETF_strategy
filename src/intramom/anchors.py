"""前序 RUN-001（Intraday_Momentum_for_ETF 项目）已审数值锚点。

同口径重跑必须逐值一致（浮点容差 1e-9），这是新引擎忠实复现的强验收。
来源：Intraday_Momentum_for_ETF/output/08_summary_stats.csv、05/06/07_meta.json。
"""
from __future__ import annotations

# (variant, grid) → 锚点
RUN001_ANCHORS: dict[tuple[str, str], dict] = {
    ("A", "g30"): {"exec_rows": 2840, "end_aum": 1222781.49153, "TradedDays": 1375,
                   "TotalReturn": 0.22278149152999993, "Sharpe": 0.26887584501202494,
                   "MDD": -0.23285728641000325,
                   "first_fill": {"date": "2017-01-23", "decision_time": "10:00",
                                  "exec_time": "10:01", "side": "买入",
                                  "price": 3.378, "shares": 297600}},
    ("B", "g30"): {"exec_rows": 3410, "end_aum": 811179.3446699956, "TradedDays": 1375,
                   "TotalReturn": -0.18882065533000436, "Sharpe": -0.24381843776236323,
                   "MDD": -0.275856277385913},
    ("C", "g30"): {"exec_rows": 3410, "end_aum": 419148.8050699884, "TradedDays": 1375,
                   "TotalReturn": -0.5808511949300116, "Sharpe": -0.5371431677507539,
                   "MDD": -0.5827042757420573},
    ("A", "g15"): {"exec_rows": None, "end_aum": None,
                   "TotalReturn": 0.1799905794299992},
    ("A", "g05"): {"exec_rows": None, "end_aum": None,
                   "TotalReturn": 0.2514026577800046},
    ("B", "g15"): {"exec_rows": 4318, "end_aum": 535332.0999899979},
    ("B", "g05"): {"exec_rows": 6430, "end_aum": 251420.2096899998},
    ("C", "g15"): {"exec_rows": 4318, "end_aum": 169974.32535999274},
    ("C", "g05"): {"exec_rows": 6430, "end_aum": 27309.02847000057},
}

RTOL = 1e-9


def check_anchors(result, summary: dict) -> list[dict]:
    """对照 RUN-001 锚点，返回勾稽行列表（name/expected/actual/status）。"""
    cfg = result.config
    key = (cfg.variant_tag, cfg.grid)
    if key not in RUN001_ANCHORS:
        return []
    anc = RUN001_ANCHORS[key]
    rows = []

    def _cmp(name, expected, actual, int_exact=False):
        if expected is None:
            return
        if int_exact:
            ok = int(expected) == int(actual)
        else:
            ok = abs(float(actual) - float(expected)) <= RTOL * max(1.0, abs(float(expected)))
        rows.append({"name": name, "expected": expected, "actual": actual,
                     "status": "PASS" if ok else "FAIL"})

    _cmp("exec_rows", anc.get("exec_rows"), len(result.trades), int_exact=True)
    _cmp("end_aum", anc.get("end_aum"), result.end_aum)
    _cmp("TradedDays", anc.get("TradedDays"), summary.get("TradedDays"), int_exact=True)
    _cmp("TotalReturn", anc.get("TotalReturn"), summary.get("TotalReturn"))
    _cmp("Sharpe", anc.get("Sharpe"), summary.get("Sharpe"))
    _cmp("MDD", anc.get("MDD"), summary.get("MDD"))

    ff = anc.get("first_fill")
    if ff is not None and len(result.trades):
        t0 = result.trades.iloc[0]
        ok = (str(t0["date"])[:10] == ff["date"]
              and t0["decision_time"] == ff["decision_time"]
              and t0["exec_time"] == ff["exec_time"]
              and t0["side"] == ff["side"]
              and abs(t0["price"] - ff["price"]) < 1e-9
              and int(t0["shares"]) == ff["shares"])
        rows.append({"name": "first_fill",
                     "expected": f"{ff['date']} {ff['exec_time']} {ff['side']} {ff['shares']}@{ff['price']}",
                     "actual": f"{str(t0['date'])[:10]} {t0['exec_time']} {t0['side']} {int(t0['shares'])}@{t0['price']}",
                     "status": "PASS" if ok else "FAIL"})
    return rows


def anchors_pass(rows: list[dict]) -> bool:
    return bool(rows) and all(r["status"] == "PASS" for r in rows)
