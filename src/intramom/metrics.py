"""绩效统计：汇总指标（论文 Tables 1/2/3/A1 口径）、整趟统计（Table 4）、alpha/beta 回归。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def perf_stats(ret, equity, init: float = 1_000_000.0) -> dict:
    """日收益序列 → 汇总指标（rf=0；IRR=(1+TR)^(252/N)−1；MDD 基于日净值）。"""
    r = np.asarray(ret, dtype=float)
    eq = np.asarray(equity, dtype=float)
    n = len(r)
    total = eq[-1] / init - 1.0
    irr = (1.0 + total) ** (252.0 / n) - 1.0 if (n > 0 and total > -1) else np.nan
    sd = r.std(ddof=1) if n > 1 else np.nan
    vol = sd * math.sqrt(252.0)
    sharpe = r.mean() / sd * math.sqrt(252.0) if (sd and sd > 0) else np.nan
    hit = float((r > 0).mean()) if n else np.nan
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min()) if n else np.nan
    skew = float(pd.Series(r).skew()) if n > 2 else np.nan
    return {"N": n, "TotalReturn": total, "IRR": irr, "Vol": vol, "Sharpe": sharpe,
            "HitRatio": hit, "MDD": mdd, "Skew": skew,
            "WorstDay": float(r.min()) if n else np.nan,
            "BestDay": float(r.max()) if n else np.nan,
            "AvgDailyBps": float(r.mean()) * 1e4 if n else np.nan}


def ols_multi(y, X: np.ndarray, names: list[str]) -> dict:
    """多元 OLS（含常数列时需自行在 X 第一列放 1）：返回各系数/t/p 与 R²。

    X: n × k 矩阵；names: 各列名称。返回 {"coef": {name: v}, "t": {...}, "p": {...}, "n", "r2"}。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    X, y = X[m], y[m]
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = float(resid @ resid) / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    tvals = beta / se
    pvals = [math.erfc(abs(t) / math.sqrt(2.0)) for t in tvals]
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return {"coef": {nm: float(b) for nm, b in zip(names, beta)},
            "t": {nm: float(t) for nm, t in zip(names, tvals)},
            "p": {nm: float(p) for nm, p in zip(names, pvals)},
            "n": int(n), "r2": r2}


def ols_stats(y, x) -> dict:
    """日频 OLS：y = α + β·x + ε（α 日频，年化 ×252）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = float(resid @ resid) / (n - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    tvals = beta / se
    pvals = [math.erfc(abs(t) / math.sqrt(2.0)) for t in tvals]
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return {"alpha": float(beta[0]), "beta": float(beta[1]),
            "alpha_annual": float(beta[0]) * 252.0,
            "t_alpha": float(tvals[0]), "t_beta": float(tvals[1]),
            "p_alpha": float(pvals[0]), "p_beta": float(pvals[1]), "n": n, "r2": r2}


def summary_row(result, traded_col: str = "traded") -> dict:
    """单次回测 → 一行汇总（Variant/Grid/N/TradedDays/TotalReturn/.../AvgDailyBps）。"""
    cfg = result.config
    dl = result.daily
    st = perf_stats(dl["ret"].to_numpy(), dl["aum_end"].to_numpy(),
                    init=cfg.initial_capital)
    traded = dl[dl[traded_col] == 1]
    st["HitRatioTraded"] = float((traded["ret"] > 0).mean()) if len(traded) else np.nan
    st["TradedDays"] = int(len(traded))
    st["Variant"] = cfg.variant_tag
    st["Grid"] = cfg.grid
    st["Run"] = cfg.name
    cols = ["Run", "Variant", "Grid", "N", "TradedDays", "TotalReturn", "IRR", "Vol",
            "Sharpe", "HitRatio", "HitRatioTraded", "MDD", "Skew",
            "WorstDay", "BestDay", "AvgDailyBps"]
    return {c: st[c] for c in cols}


def round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """fills → 整趟（开→平）；PnL/Share = 整趟净盈亏/份额（含成本）。"""
    rts = []
    for tid, grp in trades.groupby("trade_id"):
        grp = grp.sort_index()
        op = grp[grp.action.str.startswith("OPEN")]
        cl = grp[grp.action.str.startswith(("CLOSE", "LADDER"))]  # 阶梯止盈腿计入平仓
        if len(op) != 1 or len(cl) < 1:
            raise RuntimeError(f"整趟异常 trade_id={tid}")
        o = op.iloc[0]
        c = cl[cl.action.str.startswith("CLOSE")].iloc[-1]  # 退出时点=最后一笔 CLOSE
        direction = 1 if o.action == "OPEN_LONG" else -1
        # 分腿计算（兼容阶梯止盈的多次平仓）：gross = Σ dir×(平仓价−开仓价)×该腿份额
        gross = float(sum(direction * (cl_r["price"] - o["price"]) * cl_r["shares"]
                          for _, cl_r in cl.iterrows()))
        costs = grp["cost"].sum()
        pnl = gross - costs
        rts.append({"trade_id": tid, "date": o["date"],
                    "direction": "long" if direction == 1 else "short",
                    "entry_time": o["decision_time"], "exit_time": c["decision_time"],
                    "entry_px": o["price"], "exit_px": c["price"], "shares": o["shares"],
                    "gross": gross, "costs": costs, "pnl": pnl,
                    "pnl_per_share": pnl / o["shares"],
                    "ret_on_notional": pnl / (o["price"] * o["shares"]),
                    "n_exec": len(grp)})
    return pd.DataFrame(rts)


def trade_level_stats(rt: pd.DataFrame, n_days: int) -> dict:
    """整趟 → Table 4 口径交易级统计。"""
    worst = rt.loc[rt["ret_on_notional"].idxmin()]
    best = rt.loc[rt["ret_on_notional"].idxmax()]
    return {"Trades": len(rt),
            "AvgTradePerDay": len(rt) / n_days,
            "HitRatio": float((rt["pnl"] > 0).mean()),
            "AvgPnLPerShare": float(rt["pnl_per_share"].mean()),
            "AvgRetOnNotionalBps": float(rt["ret_on_notional"].mean()) * 1e4,
            "MaxLossTradePct": float(worst["ret_on_notional"]) * 100,
            "MaxLossDate": str(pd.Timestamp(worst["date"]).date()),
            "MaxGainTradePct": float(best["ret_on_notional"]) * 100,
            "MaxGainDate": str(pd.Timestamp(best["date"]).date()),
            "LongShare": float((rt["direction"] == "long").mean())}
