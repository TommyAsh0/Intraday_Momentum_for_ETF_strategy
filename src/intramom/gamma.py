"""GAMMA 状态（GEX）面板：由期权日行情反解 IV → BS gamma → 日度 GEX 序列。

口径（做市商惯例，SqueezeMetrics 风格）：
- 假设客户净买 put（保护）/ 净卖 call（备兑）→ 做市商 long call + short put
- 合约 GEX = sign × OI × Γ × per_unit × S² × 0.01，call 取 +1、put 取 −1（元 / 1% 标的变动）
- 日度 GEX(t) = 当日全部存活合约 GEX 之和；GEX>0 = 做市商 long gamma（压波动），
  GEX<0 = short gamma（助趋势）
- IV 反解：BS（含连续分红率 q 近似，q = 过去 365 日累计每股分红 / 当前价），二分法；
  利率 r 常数 2%（正负区间对该参数不敏感）
- 合约过滤：settle>0、oi 非空、T ≥ 2/365、IV ∈ (1e-4, 5]

无前视对齐：GEX(t) 用 t 日收盘 settle/oi 计算 → 分析 t+1 日策略收益时 shift(1)。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import ndtr

R_FREE = 0.02          # 常数无风险利率
MIN_T = 2.0 / 365.0    # 剔除到期前不足 2 日的合约（gamma 爆炸且结算扰动）
IV_BOUNDS = (1e-4, 5.0)


# ---------------------------------------------------------------- BS 模型
def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return ndtr(x)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(S, K, T, r, q, sigma, cp) -> np.ndarray:
    """BS 价格（连续分红率 q）。cp: +1 call / -1 put。向量化。"""
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma), )
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    return cp * (S * df_q * _norm_cdf(cp * d1) - K * df_r * _norm_cdf(cp * d2))


def bs_gamma(S, K, T, r, q, sigma) -> np.ndarray:
    """BS gamma（call/put 相同）。向量化。"""
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sq
    return np.exp(-q * T) * _norm_pdf(d1) / (S * sq)


def implied_vol(price, S, K, T, r, q, cp,
                lo: float = IV_BOUNDS[0], hi: float = IV_BOUNDS[1],
                tol: float = 1e-10, max_iter: int = 100) -> np.ndarray:
    """二分法反解 IV（向量化）；无界内解返回 NaN。"""
    price = np.asarray(price, dtype=float)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    cp = np.asarray(cp, dtype=float)
    iv = np.full(price.shape, np.nan)
    p_lo = bs_price(S, K, T, r, q, np.full_like(price, lo), cp)
    p_hi = bs_price(S, K, T, r, q, np.full_like(price, hi), cp)
    ok = (price >= p_lo - 1e-12) & (price <= p_hi + 1e-12) & (T > 0) & (S > 0) & (K > 0)
    a = np.where(ok, lo, np.nan)
    b = np.where(ok, hi, np.nan)
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        pm = bs_price(S, K, T, r, q, mid, cp)
        go_hi = pm < price
        a = np.where(go_hi, mid, a)
        b = np.where(go_hi, b, mid)
        if np.nanmax(np.abs(pm - price)) < tol:
            break
    iv = 0.5 * (a + b)
    iv[~np.isfinite(iv)] = np.nan
    return iv


# ---------------------------------------------------------------- 分红率
def dividend_yield_series(exdiv: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """q(t) = 过去 365 日累计每股分红 / 当日收盘（前视安全：除息日及以后计入）。"""
    d = daily.sort_values("trade_date").copy()
    d["div"] = 0.0
    ex = exdiv.copy()
    ex["trade_date"] = pd.to_datetime(ex["trade_date"])
    ex = ex.set_index("trade_date")["div_implied"]
    d = d.set_index("trade_date")
    d.loc[d.index.isin(ex.index), "div"] = ex
    trailing = d["div"].rolling(365, min_periods=0).sum()
    q = trailing / d["close"]
    return q.clip(lower=0.0).fillna(0.0)


# ---------------------------------------------------------------- GEX 面板
def build_gex_daily(opt_basic: pd.DataFrame, opt_daily: pd.DataFrame,
                    daily_etf: pd.DataFrame, exdiv: pd.DataFrame,
                    r: float = R_FREE) -> tuple[pd.DataFrame, dict]:
    """日度 GEX 面板。返回 (gex_daily, qa)。

    gex_daily 列：date, S, n_contracts, oi_sum, iv_median, gex_call, gex_put, gex,
    gex_yi（亿元/1%）。
    """
    b = opt_basic.copy()
    d = opt_daily.merge(
        b[["ts_code", "call_put", "exercise_price", "maturity_date", "per_unit"]],
        on="ts_code", how="left")
    d["trade_date"] = pd.to_datetime(d["trade_date"], format="%Y%m%d")
    d["maturity"] = pd.to_datetime(d["maturity_date"], format="%Y%m%d")
    d["T"] = (d["maturity"] - d["trade_date"]).dt.days / 365.0

    px = daily_etf.set_index("trade_date")["close"]
    q = dividend_yield_series(exdiv, daily_etf)
    d["S"] = d["trade_date"].map(px)
    d["q"] = d["trade_date"].map(q).fillna(0.0)
    d["cp"] = (d["call_put"] == "C").map({True: 1.0, False: -1.0})

    # 过滤 + IV 反解
    valid = (d["settle"] > 0) & d["oi"].notna() & (d["T"] >= MIN_T) & d["S"].notna()
    dv = d[valid].copy()
    dv["iv"] = implied_vol(dv["settle"].to_numpy(), dv["S"].to_numpy(),
                           dv["exercise_price"].to_numpy(), dv["T"].to_numpy(),
                           r, dv["q"].to_numpy(), dv["cp"].to_numpy())
    iv_ok = dv["iv"].notna() & (dv["iv"] > IV_BOUNDS[0]) & (dv["iv"] <= IV_BOUNDS[1])
    dv = dv[iv_ok].copy()
    dv["gamma"] = bs_gamma(dv["S"].to_numpy(), dv["exercise_price"].to_numpy(),
                           dv["T"].to_numpy(), r, dv["q"].to_numpy(), dv["iv"].to_numpy())
    # GEX（元 / 1% 标的变动）：call + / put −
    dv["gex"] = dv["cp"] * dv["oi"] * dv["gamma"] * dv["per_unit"] * dv["S"] ** 2 * 0.01
    # 变体权重（日内归一：当日权重均值 = 1，保持与基准 GEX 同量纲可比）
    # - IV 加权：高 IV 合约（通常更靠近平值/更受关注）权重更大
    dv["w_iv"] = dv["iv"] / dv.groupby("trade_date")["iv"].transform("mean")
    # - 近月加权：T 越小权重越大（τ=45 日指数衰减），近月 gamma 对做市商对冲压力主导
    TAU = 45.0 / 365.0
    dv["w_near"] = np.exp(-dv["T"] / TAU)
    dv["w_near"] = dv["w_near"] / dv.groupby("trade_date")["w_near"].transform("mean")
    dv["gex_ivw"] = dv["gex"] * dv["w_iv"]
    dv["gex_near"] = dv["gex"] * dv["w_near"]

    g = dv.groupby("trade_date").agg(
        S=("S", "first"), n_contracts=("ts_code", "size"), oi_sum=("oi", "sum"),
        iv_median=("iv", "median"),
        gex_call=("gex", lambda s: s[s > 0].sum()),
        gex_put=("gex", lambda s: s[s < 0].sum()),
        gex=("gex", "sum"),
        gex_ivw=("gex_ivw", "sum"),
        gex_near=("gex_near", "sum"),
    ).reset_index().rename(columns={"trade_date": "date"})
    g["gex_yi"] = g["gex"] / 1e8
    g["gex_ivw_yi"] = g["gex_ivw"] / 1e8
    g["gex_near_yi"] = g["gex_near"] / 1e8

    qa = {
        "rows_raw": int(len(d)), "rows_valid_filters": int(valid.sum()),
        "rows_iv_ok": int(len(dv)),
        "iv_fail_share": float(1 - len(dv) / max(valid.sum(), 1)),
        "n_days": int(len(g)),
        "date_range": [str(g["date"].min().date()), str(g["date"].max().date())],
        "gex_pos_days": int((g["gex"] > 0).sum()),
        "gex_neg_days": int((g["gex"] < 0).sum()),
        "gex_yi_range": [float(g["gex_yi"].min()), float(g["gex_yi"].max())],
    }
    return g, qa
