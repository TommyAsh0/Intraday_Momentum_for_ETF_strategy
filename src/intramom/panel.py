"""面板预处理：把分钟/日频数据一次性矩阵化，供回测引擎与多线程遍历只读共享。

面板内容（全部 days × labels 的 numpy 矩阵 + 引擎热路径用的 Python list 版本）：
- 价格矩阵 O_m/C_m/VOL_m（241 列：09:30 竞价 + 240 个 1 分钟价格标签）
- 信号价/成交价由 grid_spec 给出列号：信号 = 标签 τ bar 收盘，成交 = 下一分钟 bar 开盘
- MOVE 矩阵（|Close(τ)/Open−1|，240 价格标签）→ sigma_panel(lookback) 滚动均值(shift 1，无前视)
- 日频 open/pre_close/ret/σ_daily（动态仓位杠杆用）
- 各决策网格 VWAP 面板（决策间隔 bar 聚合，日内累计，无前视）
- 可买/可卖门禁（涨跌停封板 + 零成交分钟）

σ 面板 / VWAP 面板 / σ_daily 均按参数惰性计算并缓存——参数遍历时不同 lookback/网格
共享同一份不可变底稿，这是"面板预处理 + 多线程拟合交易"的核心。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_loader import EXPECTED_LABELS, SIG_BAR, DataBundle


@dataclass
class PanelData:
    """不可变回测底稿（构建一次，跨线程只读共享；缓存字典仅在构建进程内填充）。"""

    days: pd.DatetimeIndex
    labels: list[str]                       # 241 个 bar 标签
    price_labels: list[str]                 # 240 个价格标签（09:31..15:00）
    idx: dict[str, int]                     # 标签 → 列号
    ppos: dict[str, int]                    # 价格标签 → move/σ 矩阵列号
    # numpy 矩阵
    O_m: np.ndarray
    C_m: np.ndarray
    VOL_m: np.ndarray
    H_m: np.ndarray                           # 持仓路径研究用（MAE/MFE），引擎不使用
    L_m: np.ndarray
    AMT_m: np.ndarray                         # 成交额（元），1 分钟 VWAP 用
    move: np.ndarray                        # |Close(τ)/Open − 1|，days × 240
    day_open: np.ndarray
    pre_close_v: np.ndarray
    ret_v: np.ndarray
    bh_nav: np.ndarray
    buyable: np.ndarray                     # days × 241 bool
    sellable: np.ndarray
    daily_close: np.ndarray                 # 分钟聚合 15:00 收盘
    daily_df: pd.DataFrame                  # 日频汇总（date/open/close/pre_close/ret/bh_nav）
    # 引擎热路径 list 版本
    C_l: list = field(default=None)
    O_l: list = field(default=None)
    H_l: list = field(default=None)
    L_l: list = field(default=None)
    BUY_l: list = field(default=None)
    SELL_l: list = field(default=None)
    do_l: list = field(default=None)
    pc_l: list = field(default=None)
    # 参数化缓存
    _sigma_cache: dict = field(default_factory=dict)
    _sigma_l_cache: dict = field(default_factory=dict)
    _sigma_daily_cache: dict = field(default_factory=dict)
    _vwap_cache: dict = field(default_factory=dict)
    _vwap_l_cache: dict = field(default_factory=dict)
    _vwap1m_cache: list = field(default=None, repr=False)  # 1 分钟 VWAP（list 版本）
    _vwap5_cache: list = field(default=None, repr=False)   # 信号后 5 分钟 VWAP（list 版本）
    _spec_cache: dict = field(default_factory=dict)
    _dl_full: pd.DataFrame = field(default=None, repr=False)  # σ_daily 源（全历史日线）
    _idx_daily: pd.DataFrame = field(default=None, repr=False)  # 指数日线（vol_source="index"）

    # ------------------------------------------------------------ 构建
    @classmethod
    def build(cls, bundle: DataBundle) -> "PanelData":
        mn = bundle.minute
        labels = list(EXPECTED_LABELS)
        idx = {s: i for i, s in enumerate(labels)}
        price_labels = labels[1:]
        ppos = {s: i for i, s in enumerate(price_labels)}

        days = pd.DatetimeIndex(sorted(mn.trade_date.unique()))
        n_days, n_lab = len(days), len(labels)

        def to_mat(col: str) -> np.ndarray:
            piv = mn.pivot(index="trade_date", columns="hhmm", values=col)
            piv = piv.reindex(index=days, columns=labels)
            return piv.to_numpy(dtype=float)

        O_m, C_m, VOL_m = to_mat("open"), to_mat("close"), to_mat("vol")
        H_m, L_m = to_mat("high"), to_mat("low")  # MAE/MFE 等持仓路径研究用（引擎不使用）
        AMT_m = to_mat("amount")                # 元（1 分钟 VWAP 用）
        i1500 = idx["15:00"]
        if np.isnan(C_m[:, i1500]).any():
            raise AssertionError("存在缺失 15:00 收盘的交易日")

        # 日 Open = 09:30 竞价 bar 开盘；2017-08-09 缺竞价 bar → 当日首根有效 bar 开盘
        i0930 = idx["09:30"]
        first_valid = np.argmax(~np.isnan(O_m), axis=1)
        day_open = np.where(np.isnan(O_m[:, i0930]),
                            O_m[np.arange(n_days), first_valid], O_m[:, i0930])

        # 日频序列（σ_daily 预热依赖样本前的日频历史，故用全历史日线计算）
        dl_full = bundle.daily.sort_values("trade_date").reset_index(drop=True)
        dl_full["ret"] = dl_full["pct_chg"] / 100.0
        s = dl_full.set_index("trade_date")
        pre_close_v = s["pre_close"].reindex(days).to_numpy(dtype=float)
        ret_v = s["ret"].reindex(days).to_numpy(dtype=float)
        if np.isnan(pre_close_v).any() or np.isnan(ret_v).any():
            raise AssertionError("日频 pre_close/ret 存在缺失")
        daily_close = C_m[:, i1500]
        bh_nav = np.cumprod(1.0 + ret_v)

        # 可交易性门禁：封板不可买/卖；零成交分钟双边封锁
        trad = bundle.tradability
        buyable = np.ones((n_days, n_lab), dtype=bool)
        sellable = np.ones((n_days, n_lab), dtype=bool)
        day_pos = {d: i for i, d in enumerate(days)}
        trad_in = trad[trad.trade_date.isin(days)]
        for r in trad_in[trad_in["可买"] == 0].itertuples():
            buyable[day_pos[r.trade_date], idx[r.label]] = False
        for r in trad_in[trad_in["可卖"] == 0].itertuples():
            sellable[day_pos[r.trade_date], idx[r.label]] = False
        tz = trad_in[trad_in["是否有成交"] == 0]
        for r in tz.itertuples():
            buyable[day_pos[r.trade_date], idx[r.label]] = False
            sellable[day_pos[r.trade_date], idx[r.label]] = False

        daily_df = pd.DataFrame({
            "date": days, "open": day_open, "close": daily_close,
            "pre_close": pre_close_v, "ret": ret_v, "bh_nav": bh_nav,
        })

        panel = cls(
            days=days, labels=labels, price_labels=price_labels, idx=idx, ppos=ppos,
            O_m=O_m, C_m=C_m, VOL_m=VOL_m, H_m=H_m, L_m=L_m, AMT_m=AMT_m,
            move=np.abs(C_m[:, 1:] / day_open[:, None] - 1.0),
            day_open=day_open, pre_close_v=pre_close_v, ret_v=ret_v, bh_nav=bh_nav,
            buyable=buyable, sellable=sellable,
            daily_close=daily_close, daily_df=daily_df,
            C_l=C_m.tolist(), O_l=O_m.tolist(), H_l=H_m.tolist(), L_l=L_m.tolist(),
            BUY_l=buyable.tolist(), SELL_l=sellable.tolist(),
            do_l=day_open.tolist(), pc_l=pre_close_v.tolist(),
        )
        panel._dl_full = dl_full  # σ_daily 计算的源（全历史）
        # 指数日收益序列（vol_source="index" 用；ext/scan 标的 index_daily=daily 占位，
        # 即退化为 ETF 自身口径）
        idx_d = bundle.index_daily.sort_values("trade_date").reset_index(drop=True).copy()
        if "pct_chg" in idx_d.columns:
            idx_d["ret"] = idx_d["pct_chg"] / 100.0
        else:
            idx_d["ret"] = idx_d["close"].pct_change()
        panel._idx_daily = idx_d.dropna(subset=["ret"])  # 保留全列（stop_rate 等研究用 open/close）
        return panel

    # ------------------------------------------------------------ 网格规格
    def grid_spec(self, taus: list[str] | tuple[str, ...]) -> list[tuple[str, int, int, int]]:
        """每个决策时点 → (tau, 信号列 sc, 成交列 ec, move/σ 列 pi)。

        信号价 = 标签 τ bar 收盘（13:00 用 11:30 bar）；成交价 = 下一分钟 bar 开盘。
        """
        key = tuple(taus)
        if key in self._spec_cache:
            return self._spec_cache[key]
        spec = []
        for tau in taus:
            sbar = SIG_BAR.get(tau, tau)
            if sbar not in self.idx:
                raise ValueError(f"决策时点 {tau} 无对应 bar 标签")
            sc = self.idx[sbar]
            if sc + 1 >= len(self.labels):
                raise ValueError(f"决策时点 {tau} 之后无成交 bar（最后一个合法时点为 14:59 前）")
            spec.append((tau, sc, sc + 1, self.ppos[sbar]))
        self._spec_cache[key] = spec
        return spec

    # ------------------------------------------------------------ σ 面板（噪声带核心）
    def sigma_panel(self, lookback: int) -> np.ndarray:
        """σ(t,τ) = 前 lookback 个交易日 move 列均值（shift 1，t 日开盘前可得）。"""
        if lookback not in self._sigma_cache:
            sg = pd.DataFrame(self.move).rolling(lookback).mean().shift(1).to_numpy()
            if not np.isnan(sg[:lookback]).all():
                raise AssertionError("σ 面板预热期应全为 NaN")
            if np.isnan(sg[lookback:]).any():
                raise AssertionError("σ 面板预热期后存在 NaN（move 矩阵有缺口）")
            self._sigma_cache[lookback] = sg
            self._sigma_l_cache[lookback] = sg.tolist()
        return self._sigma_cache[lookback]

    def sigma_panel_l(self, lookback: int) -> list:
        self.sigma_panel(lookback)
        return self._sigma_l_cache[lookback]

    # ------------------------------------------------------------ 日频波动（动态仓位）
    def sigma_daily(self, vol_lookback: int, source: str = "etf") -> np.ndarray:
        """σ_daily,t = 前 vol_lookback 个交易日日收益样本 std（ddof=1，shift 1）。

        source: "etf"=所交易 ETF 自身日收益（论文口径 σ_SPY,t）；
                "index"=对标指数日收益（000300.SH，价格指数不含分红）。
        """
        if source not in ("etf", "index"):
            raise ValueError(f"vol_source 必须为 etf/index，得到 {source!r}")
        key = (vol_lookback, source)
        if key not in self._sigma_daily_cache:
            if source == "index" and self._idx_daily is not None:
                s = self._idx_daily.set_index("trade_date")["ret"]
            else:
                s = self._dl_full.set_index("trade_date")["ret"]
            sig = s.rolling(vol_lookback).std(ddof=1).shift(1)
            v = sig.reindex(self.days).to_numpy(dtype=float)
            # 上市晚于样本起点的标的（如 159845 2021-03 上市）：允许开头预热段 NaN
            # （引擎对该段不交易），但首个有效值之后不得再有缺失
            first_valid = np.argmax(~np.isnan(v)) if (~np.isnan(v)).any() else len(v)
            if first_valid == len(v) or np.isnan(v[first_valid:]).any():
                raise AssertionError("σ_daily 预热完成后仍存在缺失")
            self._sigma_daily_cache[key] = v
        return self._sigma_daily_cache[key]

    # ------------------------------------------------------------ VWAP 面板
    def vwap_panel(self, taus: list[str] | tuple[str, ...]) -> np.ndarray:
        """按决策间隔 bar 聚合的日内累计 VWAP（仅用 ≤τ 数据，无前视）。

        网格 bar =（前一信号标签, 当前信号标签] 的 1 分钟 bar 合计（首根自 09:30 含竞价 bar 起；
        13:00 决策点用 11:30 bar；午休无 bar 自然跳过；零成交 bar 不产生增量）。
        返回 days × 241 矩阵，仅信号列有值。
        """
        key = tuple(taus)
        if key in self._vwap_cache:
            return self._vwap_cache[key]
        sigs = [SIG_BAR.get(t, t) for t in taus] + ["15:00"]
        cols, c0 = [], 0
        for slab in sigs:
            c1 = self.idx[slab]
            cols.append((slab, c0, c1))
            c0 = c1 + 1
        n_days = len(self.days)
        pv = np.zeros((n_days, len(cols)))
        vv = np.zeros((n_days, len(cols)))
        for k, (slab, c0, c1) in enumerate(cols):
            close_k = self.C_m[:, c1]
            vol_k = np.nansum(np.where(np.isnan(self.VOL_m[:, c0:c1 + 1]), 0.0,
                                       self.VOL_m[:, c0:c1 + 1]), axis=1)
            bad = np.isnan(close_k)  # 缺口 bar（2017-08-09 缺竞价 bar）该根贡献置零
            pv[:, k] = np.where(bad, 0.0, close_k * vol_k)
            vv[:, k] = np.where(bad, 0.0, vol_k)
        cs_pv = np.cumsum(pv, axis=1)
        cs_vv = np.cumsum(vv, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            vw = np.where(cs_vv > 0, cs_pv / cs_vv, np.nan)
        arr = np.full((n_days, len(self.labels)), np.nan)
        for k, (slab, c0, c1) in enumerate(cols):
            arr[:, c1] = vw[:, k]
        if np.isnan(arr[:, [c1 for _, _, c1 in cols]]).any():
            raise AssertionError("VWAP 信号列存在 NaN（整日零成交？）")
        self._vwap_cache[key] = arr
        self._vwap_l_cache[key] = arr.tolist()
        return arr

    def vwap_panel_l(self, taus: list[str] | tuple[str, ...]) -> list:
        self.vwap_panel(taus)
        return self._vwap_l_cache[tuple(taus)]

    # ------------------------------------------------------------ 1 分钟 VWAP（L-1 止损用）
    def vwap_1min_l(self) -> list:
        """标准日内分钟 VWAP：累计 amount / (累计 vol × 100)（零成交 bar 不产生增量）。

        与 vwap_panel（决策间隔 bar 聚合口径，DEC-009）不同：这是每分钟全分辨率口径，
        仅用于 stop_freq='1min'（L-1 变体）的每分钟止损判定。
        首次放量之前的分钟 VWAP 未定义（NaN，引擎止损回退到带）；其后前向填充。
        """
        if self._vwap1m_cache is None:
            amt = np.where(np.isnan(self.AMT_m), 0.0, self.AMT_m)
            vol = np.where(np.isnan(self.VOL_m), 0.0, self.VOL_m)
            cs_a = np.cumsum(amt, axis=1)
            cs_v = np.cumsum(vol, axis=1) * 100.0  # 手 → 份
            with np.errstate(invalid="ignore", divide="ignore"):
                vw = np.where(cs_v > 0, cs_a / cs_v, np.nan)
            vw = pd.DataFrame(vw).ffill(axis=1).to_numpy()  # 首次放量后前向填充
            if np.isnan(vw[:, -1]).any():
                raise AssertionError("1 分钟 VWAP 存在整日 NaN（全天零成交？）")
            self._vwap1m_cache = vw.tolist()
        return self._vwap1m_cache

    # ------------------------------------------------------------ 信号后 5 分钟 VWAP（研报 2.0 执行口径）
    def vwap5_l(self) -> list:
        """vwap5[d, j] = 第 j 根 bar 起 5 根 bar（j..j+4，尾部自动截断至 15:00）的 VWAP。

        用于 fill_model='vwap5'：信号在 τ（bar τ 收盘）→ 成交价为 bar τ+1..τ+5 的 VWAP；
        EOD 平仓取 vwap5[., 236]（14:56..15:00 五根，即"收盘前 5 分钟 VWAP"）。
        零成交 bar 不产生增量；窗口内全零成交则回退该窗口末端 bar 收盘价。
        """
        if self._vwap5_cache is None:
            amt = np.where(np.isnan(self.AMT_m), 0.0, self.AMT_m)
            vol = np.where(np.isnan(self.VOL_m), 0.0, self.VOL_m) * 100.0  # 手 → 份
            cs_a = np.concatenate([np.zeros((self.n_days, 1)),
                                   np.cumsum(amt, axis=1)], axis=1)
            cs_v = np.concatenate([np.zeros((self.n_days, 1)),
                                   np.cumsum(vol, axis=1)], axis=1)
            n = amt.shape[1]
            out = np.empty_like(amt)
            for j in range(n):
                j2 = min(j + 5, n)
                a = cs_a[:, j2] - cs_a[:, j]
                v = cs_v[:, j2] - cs_v[:, j]
                with np.errstate(invalid="ignore", divide="ignore"):
                    w = np.where(v > 0, a / v, np.nan)
                fallback = self.C_m[:, j2 - 1]  # 全零成交窗口回退收盘
                out[:, j] = np.where(np.isnan(w), fallback, w)
            self._vwap5_cache = out.tolist()
        return self._vwap5_cache

    # ------------------------------------------------------------ 噪声带合成
    def bands(self, di: int, vm: float, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        """UB/LB（240 价格标签）：max/min(Open_t, preClose_t) × (1 ± VM·σ)。"""
        sg = self.sigma_panel(lookback)[di]
        base_up = self.day_open[di] if self.day_open[di] > self.pre_close_v[di] else self.pre_close_v[di]
        base_dn = self.day_open[di] if self.day_open[di] < self.pre_close_v[di] else self.pre_close_v[di]
        return base_up * (1.0 + vm * sg), base_dn * (1.0 - vm * sg)

    @property
    def n_days(self) -> int:
        return len(self.days)

    @property
    def i1500(self) -> int:
        return self.idx["15:00"]
