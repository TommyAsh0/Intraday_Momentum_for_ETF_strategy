"""回测引擎：逐日状态机（空仓/多/空），论文口径的噪声带突破入场 + 参数化止损 + 15:00 强制平仓。

时序纪律（无前视）：决策时点 τ 的信号价 = 标签 τ bar 收盘（13:00 用 11:30 bar）；
成交价 = 下一分钟 bar 开盘（13:00 → 13:01）；15:00 平仓 = 15:00 bar 收盘价。

规则（与前序 RUN-001 引擎逐值一致）：
- 空仓时价格新跨越 UB → 开多；新跨越 LB → 开空（armed 机制：止损/反手后不原地再入，
  需价格先回到带内重新武装）
- stop_mode：opp_band=反向带；band_vwap=多 max(UB,VWAP) / 空 min(LB,VWAP)；vwap_only=仅 VWAP
- 止损触发时若同时穿越反向带 → 平仓 + 反手（同一时点两笔成交）
- sizing：fixed=100% 仓位；vol_target=lev = min(max_leverage, σ_target/σ_daily,t)
- 份额 = floor(AUM × lev / 当日开盘 / lot) × lot；成本 = 佣金 rate×成交额 + 滑点 tick×份额，双边
- AUM 逐日递推：日收益 = 当日净盈亏(含成本)/期初 AUM；权益 ≤ 0 抛错
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .panel import PanelData

TICK = 0.001  # ETF 最小价位（元/份）——passive_chase 挂单的限价偏移

TRADE_COLS = ["date", "decision_time", "exec_time", "side", "price", "shares",
              "commission", "slippage_cost", "trade_id", "action", "reason", "lev"]
DAILY_COLS = ["date", "aum_start", "aum_end", "ret", "n_exec", "lev", "traded"]
DECISION_COLS = ["date", "tau", "sig_bar", "exec_bar", "price", "sigma", "UB", "LB",
                 "vwap", "pos_before", "pos_after", "actions", "exec_price"]


@dataclass
class BacktestResult:
    """单次回测全部产物。"""
    trades: pd.DataFrame      # 每笔成交（fills）
    daily: pd.DataFrame       # 每交易日资金线
    equity: pd.DataFrame      # 日净值
    decisions: pd.DataFrame   # 每个决策时点的信号/带/仓位/动作（审计）
    config: StrategyConfig

    @property
    def end_aum(self) -> float:
        return float(self.daily["aum_end"].iloc[-1])


def run_backtest(panel: PanelData, cfg: StrategyConfig) -> BacktestResult:
    """对给定参数运行一次完整回测。panel 只读；本函数可安全地在多线程中并发调用。"""
    if cfg.stop_freq == "1min":
        return _run_backtest_1min_stop(panel, cfg)
    if cfg.exit_mult != 1.0 or cfg.ladder_rate > 0.0 or cfg.fill_model == "vwap5":
        raise NotImplementedError("exit_mult / ladder_rate / fill_model='vwap5' 仅在 stop_freq='1min' 路径实现")
    taus = cfg.grid_taus
    spec = panel.grid_spec(taus)
    sig_l = panel.sigma_panel_l(cfg.lookback)
    need_vwap = cfg.stop_mode in ("band_vwap", "vwap_only", "band_vwap_loose")
    vw_l = panel.vwap_panel_l(taus) if need_vwap else None
    s14_l = panel.sigma_daily(cfg.vol_lookback, cfg.vol_source).tolist() if cfg.sizing == "vol_target" else None

    dyn_lev = cfg.sizing == "vol_target"
    vm = cfg.vol_mult
    commission = cfg.commission_rate
    slippage = cfg.slippage_per_share
    lot = cfg.lot_size
    sigma_target = cfg.sigma_target
    max_lev = cfg.max_leverage
    stop_mode = cfg.stop_mode
    state_mode = cfg.entry_mode == "band_state"
    passive = cfg.fill_model == "passive_chase"
    exit_mult = cfg.exit_mult

    n_days = panel.n_days
    days = panel.days
    i1500 = panel.i1500
    labels = panel.labels
    C_l, O_l = panel.C_l, panel.O_l
    H_l, L_l = panel.H_l, panel.L_l
    BUY_l, SELL_l = panel.BUY_l, panel.SELL_l
    do_l, pc_l = panel.do_l, panel.pc_l

    trades: list[tuple] = []
    daily_rows: list[tuple] = []
    decision_rows: list[tuple] = []
    aum = cfg.initial_capital
    tid = 0
    isnan = math.isnan
    floor = math.floor

    for di in range(n_days):
        aum0 = aum
        pnl_day = 0.0
        n_tr = 0
        # ---- 当日杠杆（动态仓位）：σ_daily 缺失则当日不交易 ----
        if dyn_lev:
            s14 = s14_l[di]
            if isnan(s14) or s14 <= 0:
                daily_rows.append((days[di], aum0, aum0, 0.0, 0, np.nan, 0))
                continue
            lev = min(max_lev, sigma_target / s14)
        else:
            lev = 1.0
        do = do_l[di]
        pc = pc_l[di]
        shares = floor(aum0 * lev / do / lot) * lot
        if shares <= 0:
            daily_rows.append((days[di], aum0, aum0, 0.0, 0, lev, 0))
            continue
        base_up = do if do > pc else pc
        base_dn = do if do < pc else pc
        Crow = C_l[di]
        Orow = O_l[di]
        VWrow = vw_l[di] if need_vwap else None
        SGrow = sig_l[di]
        BUYrow = BUY_l[di]
        SELLrow = SELL_l[di]
        Hrow = H_l[di]
        Lrow = L_l[di]
        pos = 0
        entry_px = 0.0
        entry_comm = 0.0
        entry_slip = 0.0
        entry_tid = 0
        armed_l = True
        armed_s = True
        for tau, sc, ec, pi_ in spec:
            p = Crow[sc]
            if p != p:
                continue
            sg = SGrow[pi_]
            if sg != sg:
                continue
            ub = base_up * (1.0 + vm * sg)
            lb = base_dn * (1.0 - vm * sg)
            ub_x = base_up * (1.0 + vm * exit_mult * sg)
            lb_x = base_dn * (1.0 - vm * exit_mult * sg)
            if p <= ub:
                armed_l = True
            if p >= lb:
                armed_s = True
            ep = Orow[ec]
            if ep != ep:
                continue
            if passive:
                _h2, _l2, _c2 = Hrow[ec], Lrow[ec], Crow[ec]
                _lb2 = round(ep - TICK, 3)
                _ls2 = round(ep + TICK, 3)
                _epb = _lb2 if _l2 <= _lb2 else _c2
                _eps = _ls2 if _h2 >= _ls2 else _c2
            else:
                _epb = _eps = ep
            pos_before = pos
            acts: list[str] = []
            if pos == 0:
                if p > ub and (armed_l or state_mode) and BUYrow[ec]:
                    if passive: ep = _epb
                    tid += 1
                    entry_tid = tid
                    entry_comm = ep * shares * commission
                    entry_slip = shares * slippage
                    entry_px = ep
                    pos = 1
                    armed_l = False
                    trades.append((days[di], tau, labels[ec], "买入", ep, shares,
                                   entry_comm, entry_slip, tid, "OPEN_LONG", "", lev))
                    acts.append("OPEN_LONG")
                    n_tr += 1
                elif p < lb and (armed_s or state_mode) and SELLrow[ec]:
                    if passive: ep = _eps
                    tid += 1
                    entry_tid = tid
                    entry_comm = ep * shares * commission
                    entry_slip = shares * slippage
                    entry_px = ep
                    pos = -1
                    armed_s = False
                    trades.append((days[di], tau, labels[ec], "卖出", ep, shares,
                                   entry_comm, entry_slip, tid, "OPEN_SHORT", "", lev))
                    acts.append("OPEN_SHORT")
                    n_tr += 1
            elif pos == 1:
                vw = VWrow[sc] if need_vwap else math.nan
                if stop_mode == "opp_band":
                    stop = lb_x
                elif stop_mode == "vwap_only":
                    stop = vw if vw == vw else lb_x
                elif stop_mode == "band_vwap_loose":
                    stop = min(ub_x, vw) if vw == vw else ub_x
                else:  # band_vwap
                    stop = max(ub_x, vw) if vw == vw else ub_x
                if p < stop and SELLrow[ec]:
                    if passive: ep = _eps
                    exit_comm = ep * shares * commission
                    exit_slip = shares * slippage
                    pnl_day += (ep - entry_px) * shares - entry_comm - entry_slip - exit_comm - exit_slip
                    trades.append((days[di], tau, labels[ec], "卖出", ep, shares,
                                   exit_comm, exit_slip, entry_tid, "CLOSE_LONG", stop_mode, lev))
                    acts.append("CLOSE_LONG")
                    n_tr += 1
                    pos = 0
                    if p < lb and (armed_s or state_mode) and SELLrow[ec]:
                        if passive: ep = _eps
                        tid += 1
                        entry_tid = tid
                        entry_comm = ep * shares * commission
                        entry_slip = shares * slippage
                        entry_px = ep
                        pos = -1
                        armed_s = False
                        trades.append((days[di], tau, labels[ec], "卖出", ep, shares,
                                       entry_comm, entry_slip, tid, "OPEN_SHORT", "REVERSAL", lev))
                        acts.append("OPEN_SHORT")
                        n_tr += 1
            else:  # pos == -1
                vw = VWrow[sc] if need_vwap else math.nan
                if stop_mode == "opp_band":
                    stop = ub_x
                elif stop_mode == "vwap_only":
                    stop = vw if vw == vw else ub_x
                elif stop_mode == "band_vwap_loose":
                    stop = max(lb_x, vw) if vw == vw else lb_x
                else:  # band_vwap
                    stop = min(lb_x, vw) if vw == vw else lb_x
                if p > stop and BUYrow[ec]:
                    if passive: ep = _epb
                    exit_comm = ep * shares * commission
                    exit_slip = shares * slippage
                    pnl_day += (entry_px - ep) * shares - entry_comm - entry_slip - exit_comm - exit_slip
                    trades.append((days[di], tau, labels[ec], "买入", ep, shares,
                                   exit_comm, exit_slip, entry_tid, "CLOSE_SHORT", stop_mode, lev))
                    acts.append("CLOSE_SHORT")
                    n_tr += 1
                    pos = 0
                    if p > ub and (armed_l or state_mode) and BUYrow[ec]:
                        if passive: ep = _epb
                        tid += 1
                        entry_tid = tid
                        entry_comm = ep * shares * commission
                        entry_slip = shares * slippage
                        entry_px = ep
                        pos = 1
                        armed_l = False
                        trades.append((days[di], tau, labels[ec], "买入", ep, shares,
                                       entry_comm, entry_slip, tid, "OPEN_LONG", "REVERSAL", lev))
                        acts.append("OPEN_LONG")
                        n_tr += 1
            decision_rows.append((days[di], tau, labels[sc], labels[ec], p, sg, ub, lb,
                                  (VWrow[sc] if need_vwap else np.nan),
                                  pos_before, pos, "+".join(acts) if acts else "NONE",
                                  ep if acts else np.nan))
        # ---- 15:00 强制平仓 ----
        if pos != 0:
            ep = Crow[i1500]
            exit_comm = ep * shares * commission
            exit_slip = shares * slippage
            if pos == 1:
                pnl_day += (ep - entry_px) * shares - entry_comm - entry_slip - exit_comm - exit_slip
                trades.append((days[di], "15:00", "15:00", "卖出", ep, shares,
                               exit_comm, exit_slip, entry_tid, "CLOSE_LONG", "EOD", lev))
                acts = "CLOSE_LONG"
            else:
                pnl_day += (entry_px - ep) * shares - entry_comm - entry_slip - exit_comm - exit_slip
                trades.append((days[di], "15:00", "15:00", "买入", ep, shares,
                               exit_comm, exit_slip, entry_tid, "CLOSE_SHORT", "EOD", lev))
                acts = "CLOSE_SHORT"
            n_tr += 1
            decision_rows.append((days[di], "15:00", labels[i1500], labels[i1500], ep, np.nan,
                                  np.nan, np.nan, np.nan, pos, 0, acts, ep))
            pos = 0
        if aum0 + pnl_day <= 0:
            raise RuntimeError(f"权益非正: {days[di].date()} aum0={aum0:.2f} pnl={pnl_day:.2f}")
        aum = aum0 + pnl_day
        daily_rows.append((days[di], aum0, aum, pnl_day / aum0, n_tr, lev, 1 if n_tr > 0 else 0))

    tr = pd.DataFrame(trades, columns=TRADE_COLS)
    tr["cost"] = tr["commission"] + tr["slippage_cost"]
    dl = pd.DataFrame(daily_rows, columns=DAILY_COLS)
    eq = dl[["date", "aum_end"]].copy()
    eq["nav"] = eq["aum_end"] / cfg.initial_capital
    dc = pd.DataFrame(decision_rows, columns=DECISION_COLS)

    # 现金账连续性勾稽：AUM 递推 == Σ日盈亏
    chk = np.abs(dl["aum_end"].to_numpy()
                 - (cfg.initial_capital + (dl["ret"] * dl["aum_start"]).cumsum().to_numpy())).max()
    if not chk < 1e-6:
        raise AssertionError(f"现金账不连续: {chk}")
    return BacktestResult(trades=tr, daily=dl, equity=eq, decisions=dc, config=cfg)


def _run_backtest_1min_stop(panel: PanelData, cfg: StrategyConfig) -> BacktestResult:
    """L-1 变体：入场仍在决策网格判定（信号不变），止损每分钟检查一次。

    与网格路径的差异仅在止损频率：
    - 每分钟标签 m ∈ 09:31..14:59：持仓中则检查止损（信号=m bar 收盘，成交=m+1 bar 开盘）
    - 网格信号 bar（含 13:00→11:30 映射）：空仓时入场判定 + armed 再武装（同网格路径）
    - 止损只平仓不反手；反手/再入场仍须等网格时点且满足"新跨越"（armed 机制）
    - B/C 类 VWAP 止损用 1 分钟全分辨率 VWAP（amount 累计口径，见 panel.vwap_1min_l）
    """
    taus = cfg.grid_taus
    spec_grid = panel.grid_spec(taus)
    sig_l = panel.sigma_panel_l(cfg.lookback)
    need_vwap = cfg.stop_mode in ("band_vwap", "vwap_only", "band_vwap_loose")
    vw_l = panel.vwap_1min_l() if need_vwap else None
    s14_l = panel.sigma_daily(cfg.vol_lookback, cfg.vol_source).tolist() if cfg.sizing == "vol_target" else None

    grid_sc = {sc: (tau, ec) for tau, sc, ec, _ in spec_grid}  # 信号列 → (tau, 成交列)

    dyn_lev = cfg.sizing == "vol_target"
    vm = cfg.vol_mult
    commission = cfg.commission_rate
    slippage = cfg.slippage_per_share
    lot = cfg.lot_size
    sigma_target = cfg.sigma_target
    max_lev = cfg.max_leverage
    stop_mode = cfg.stop_mode
    state_mode = cfg.entry_mode == "band_state"
    passive = cfg.fill_model == "passive_chase"
    vwap5 = cfg.fill_model == "vwap5"
    exit_mult = cfg.exit_mult
    ladder_rate = cfg.ladder_rate
    ladder_frac = cfg.ladder_frac
    v5_l = panel.vwap5_l() if vwap5 else None

    n_days = panel.n_days
    days = panel.days
    i1500 = panel.i1500
    i1456 = panel.idx["14:56"]  # vwap5 模式 EOD = 最后 5 分钟 VWAP
    labels = panel.labels
    ppos = panel.ppos
    C_l, O_l = panel.C_l, panel.O_l
    H_l, L_l = panel.H_l, panel.L_l
    BUY_l, SELL_l = panel.BUY_l, panel.SELL_l
    do_l, pc_l = panel.do_l, panel.pc_l

    # 每分钟规格：(label, sc, ec, pi_)，m ∈ 09:31..14:59
    spec_min = [(lab, panel.idx[lab], panel.idx[lab] + 1, ppos[lab])
                for lab in labels[1:-1]]

    trades: list[tuple] = []
    daily_rows: list[tuple] = []
    decision_rows: list[tuple] = []
    aum = cfg.initial_capital
    tid = 0
    isnan = math.isnan
    floor = math.floor

    for di in range(n_days):
        aum0 = aum
        pnl_day = 0.0
        n_tr = 0
        if dyn_lev:
            s14 = s14_l[di]
            if isnan(s14) or s14 <= 0:
                daily_rows.append((days[di], aum0, aum0, 0.0, 0, np.nan, 0))
                continue
            lev = min(max_lev, sigma_target / s14)
        else:
            lev = 1.0
        do = do_l[di]
        pc = pc_l[di]
        shares = floor(aum0 * lev / do / lot) * lot
        if shares <= 0:
            daily_rows.append((days[di], aum0, aum0, 0.0, 0, lev, 0))
            continue
        base_up = do if do > pc else pc
        base_dn = do if do < pc else pc
        Crow = C_l[di]
        Orow = O_l[di]
        VWrow = vw_l[di] if need_vwap else None
        SGrow = sig_l[di]
        BUYrow = BUY_l[di]
        SELLrow = SELL_l[di]
        Hrow = H_l[di]
        Lrow = L_l[di]
        V5row = v5_l[di] if vwap5 else None
        pos = 0
        entry_px = 0.0
        entry_comm = 0.0
        entry_slip = 0.0
        entry_tid = 0
        rem = 0          # 阶梯止盈后的剩余份额
        ladder_done = False
        armed_l = True
        armed_s = True
        for lab, sc, ec, pi_ in spec_min:
            p = Crow[sc]
            if p != p:
                continue
            sg = SGrow[pi_]
            if sg != sg:
                continue
            ub = base_up * (1.0 + vm * sg)
            lb = base_dn * (1.0 - vm * sg)
            ub_x = base_up * (1.0 + vm * exit_mult * sg)   # 收窄退出边界（exit_mult=1 时与入场带逐位相同）
            lb_x = base_dn * (1.0 - vm * exit_mult * sg)
            grid_hit = grid_sc.get(sc)
            if grid_hit is not None:
                # 网格时点：再武装（同网格路径；无论是否持仓）
                if p <= ub:
                    armed_l = True
                if p >= lb:
                    armed_s = True
            ep = Orow[ec]
            if ep != ep:
                continue
            if vwap5:
                ep = V5row[ec]
            elif passive:
                _h2, _l2, _c2 = Hrow[ec], Lrow[ec], Crow[ec]
                _lb2 = round(ep - TICK, 3)
                _ls2 = round(ep + TICK, 3)
                _epb = _lb2 if _l2 <= _lb2 else _c2
                _eps = _ls2 if _h2 >= _ls2 else _c2
            else:
                _epb = _eps = ep
            pos_before = pos
            acts: list[str] = []
            if pos == 0:
                if grid_hit is not None:
                    tau_g, ec_g = grid_hit
                    ep_g = Orow[ec_g]
                    if ep_g == ep_g:
                        if vwap5:
                            ep_g = V5row[ec_g]
                        elif passive:
                            _h3, _l3, _c3 = Hrow[ec_g], Lrow[ec_g], Crow[ec_g]
                            _lb3 = round(ep_g - TICK, 3)
                            _ls3 = round(ep_g + TICK, 3)
                            _epb_g = _lb3 if _l3 <= _lb3 else _c3
                            _eps_g = _ls3 if _h3 >= _ls3 else _c3
                        if p > ub and (armed_l or state_mode) and BUYrow[ec_g]:
                            if passive: ep_g = _epb_g
                            tid += 1
                            entry_tid = tid
                            entry_comm = ep_g * shares * commission
                            entry_slip = shares * slippage
                            entry_px = ep_g
                            pos = 1
                            rem = shares
                            ladder_done = False
                            armed_l = False
                            trades.append((days[di], tau_g, labels[ec_g], "买入", ep_g, shares,
                                           entry_comm, entry_slip, tid, "OPEN_LONG", "", lev))
                            acts.append("OPEN_LONG")
                            n_tr += 1
                        elif p < lb and (armed_s or state_mode) and SELLrow[ec_g]:
                            if passive: ep_g = _eps_g
                            tid += 1
                            entry_tid = tid
                            entry_comm = ep_g * shares * commission
                            entry_slip = shares * slippage
                            entry_px = ep_g
                            pos = -1
                            rem = shares
                            ladder_done = False
                            armed_s = False
                            trades.append((days[di], tau_g, labels[ec_g], "卖出", ep_g, shares,
                                           entry_comm, entry_slip, tid, "OPEN_SHORT", "", lev))
                            acts.append("OPEN_SHORT")
                            n_tr += 1
            elif pos == 1:
                vw = VWrow[sc] if need_vwap else math.nan
                if stop_mode == "opp_band":
                    stop = lb_x
                elif stop_mode == "vwap_only":
                    stop = vw if vw == vw else lb_x
                elif stop_mode == "band_vwap_loose":
                    stop = min(ub_x, vw) if vw == vw else ub_x
                else:  # band_vwap
                    stop = max(ub_x, vw) if vw == vw else ub_x
                # 阶梯止盈：浮盈 > ladder_rate 时先平 ladder_frac 仓位锁定（每趟一次）
                if ladder_rate > 0 and not ladder_done and (p / entry_px - 1.0) > ladder_rate and SELLrow[ec]:
                    ls = floor(shares * ladder_frac / lot) * lot
                    if 0 < ls < shares:
                        ex_comm = ep * ls * commission
                        ex_slip = ls * slippage
                        pnl_day += ((ep - entry_px) * ls
                                    - (entry_comm + entry_slip) * (ls / shares) - ex_comm - ex_slip)
                        trades.append((days[di], lab, labels[ec], "卖出", ep, ls,
                                       ex_comm, ex_slip, entry_tid, "LADDER_LONG", "ladder", lev))
                        acts.append("LADDER_LONG")
                        n_tr += 1
                        rem = shares - ls
                        ladder_done = True
                if p < stop and SELLrow[ec]:
                    if passive: ep = _eps
                    exit_comm = ep * rem * commission
                    exit_slip = rem * slippage
                    pnl_day += ((ep - entry_px) * rem
                                - (entry_comm + entry_slip) * (rem / shares) - exit_comm - exit_slip)
                    trades.append((days[di], lab, labels[ec], "卖出", ep, rem,
                                   exit_comm, exit_slip, entry_tid, "CLOSE_LONG", stop_mode, lev))
                    acts.append("CLOSE_LONG")
                    n_tr += 1
                    pos = 0
            else:  # pos == -1
                vw = VWrow[sc] if need_vwap else math.nan
                if stop_mode == "opp_band":
                    stop = ub_x
                elif stop_mode == "vwap_only":
                    stop = vw if vw == vw else ub_x
                elif stop_mode == "band_vwap_loose":
                    stop = max(lb_x, vw) if vw == vw else lb_x
                else:  # band_vwap
                    stop = min(lb_x, vw) if vw == vw else lb_x
                # 阶梯止盈（空头镜像）
                if ladder_rate > 0 and not ladder_done and (entry_px / p - 1.0) > ladder_rate and BUYrow[ec]:
                    ls = floor(shares * ladder_frac / lot) * lot
                    if 0 < ls < shares:
                        ex_comm = ep * ls * commission
                        ex_slip = ls * slippage
                        pnl_day += ((entry_px - ep) * ls
                                    - (entry_comm + entry_slip) * (ls / shares) - ex_comm - ex_slip)
                        trades.append((days[di], lab, labels[ec], "买入", ep, ls,
                                       ex_comm, ex_slip, entry_tid, "LADDER_SHORT", "ladder", lev))
                        acts.append("LADDER_SHORT")
                        n_tr += 1
                        rem = shares - ls
                        ladder_done = True
                if p > stop and BUYrow[ec]:
                    if passive: ep = _epb
                    exit_comm = ep * rem * commission
                    exit_slip = rem * slippage
                    pnl_day += ((entry_px - ep) * rem
                                - (entry_comm + entry_slip) * (rem / shares) - exit_comm - exit_slip)
                    trades.append((days[di], lab, labels[ec], "买入", ep, rem,
                                   exit_comm, exit_slip, entry_tid, "CLOSE_SHORT", stop_mode, lev))
                    acts.append("CLOSE_SHORT")
                    n_tr += 1
                    pos = 0
            if grid_hit is not None or acts:
                decision_rows.append((days[di], grid_hit[0] if grid_hit else lab,
                                      labels[sc], labels[ec], p, sg, ub, lb,
                                      (VWrow[sc] if need_vwap else np.nan),
                                      pos_before, pos, "+".join(acts) if acts else "NONE",
                                      ep if acts else np.nan))
        # ---- 15:00 强制平仓 ----
        if pos != 0:
            ep = V5row[i1456] if vwap5 else Crow[i1500]
            exit_comm = ep * rem * commission
            exit_slip = rem * slippage
            if pos == 1:
                pnl_day += ((ep - entry_px) * rem
                            - (entry_comm + entry_slip) * (rem / shares) - exit_comm - exit_slip)
                trades.append((days[di], "15:00", "15:00", "卖出", ep, rem,
                               exit_comm, exit_slip, entry_tid, "CLOSE_LONG", "EOD", lev))
                acts = "CLOSE_LONG"
            else:
                pnl_day += ((entry_px - ep) * rem
                            - (entry_comm + entry_slip) * (rem / shares) - exit_comm - exit_slip)
                trades.append((days[di], "15:00", "15:00", "买入", ep, rem,
                               exit_comm, exit_slip, entry_tid, "CLOSE_SHORT", "EOD", lev))
                acts = "CLOSE_SHORT"
            n_tr += 1
            decision_rows.append((days[di], "15:00", labels[i1500], labels[i1500], ep, np.nan,
                                  np.nan, np.nan, np.nan, pos, 0, acts, ep))
            pos = 0
        if aum0 + pnl_day <= 0:
            raise RuntimeError(f"权益非正: {days[di].date()} aum0={aum0:.2f} pnl={pnl_day:.2f}")
        aum = aum0 + pnl_day
        daily_rows.append((days[di], aum0, aum, pnl_day / aum0, n_tr, lev, 1 if n_tr > 0 else 0))

    tr = pd.DataFrame(trades, columns=TRADE_COLS)
    tr["cost"] = tr["commission"] + tr["slippage_cost"]
    dl = pd.DataFrame(daily_rows, columns=DAILY_COLS)
    eq = dl[["date", "aum_end"]].copy()
    eq["nav"] = eq["aum_end"] / cfg.initial_capital
    dc = pd.DataFrame(decision_rows, columns=DECISION_COLS)

    chk = np.abs(dl["aum_end"].to_numpy()
                 - (cfg.initial_capital + (dl["ret"] * dl["aum_start"]).cumsum().to_numpy())).max()
    if not chk < 1e-6:
        raise AssertionError(f"现金账不连续: {chk}")
    return BacktestResult(trades=tr, daily=dl, equity=eq, decisions=dc, config=cfg)
