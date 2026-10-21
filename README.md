# Intraday Momentum for ETF — 基础策略复现工程（510300.SH）

论文：*Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)*
（Zarattini, Aziz, Barbon；SSRN 4824172）的 A 股复现。

本工程在前序项目 `../Intraday_Momentum_for_ETF`（数据冻结 + notebook 全流程复现）之上，
以**工程化、参数化、可遍历**的方式重建基础策略，用于后续策略细节研究与参数遍历。

## 架构

```
面板预处理（一次构建，只读共享） ──► 参数化回测引擎（纯函数，可并发） ──► OU 风格结果目录
        PanelData                      run_backtest(panel, cfg)              data/results/<run_name>/
```

- **面板预处理**（`src/intramom/panel.py`）：分钟数据一次性矩阵化（价格/MOVE/σ 面板/VWAP/
  σ_daily/可交易性门禁），σ 与 VWAP 按参数（lookback、网格）惰性计算并缓存；
  参数遍历时所有组合共享同一份不可变底稿。
- **回测引擎**（`src/intramom/engine.py`）：逐日状态机（空仓/多/空），新跨越入场、参数化止损、
  15:00 强制平仓；信号 = τ bar 收盘 → 成交 = 下一分钟 bar 开盘，无前视。
- **多线程遍历**（`src/intramom/sweep.py`）：`ThreadPoolExecutor` 并发跑参数组合，
  主线程预填充面板缓存保证 worker 纯只读；每组合独立结果目录 + sweep 级 `summary.csv`。

## 策略规则（论文 §3 口径）

- 噪声带：`σ(t,τ) = mean_{i=1..L}|Close(τ)/Open − 1|`（L=14，shift 1 无前视）；
  `UB = max(Open, preClose)·(1+VM·σ)`，`LB = min(Open, preClose)·(1−VM·σ)`（VM=1）
- 决策网格：g30={10:00,10:30,11:00,13:00,13:30,14:00,14:30}（论文主线），另内置 g15/g05
- 入场：空仓时价格**新跨越** UB→开多 / LB→开空（armed 机制，止损后不原地再入）
- 变体：**A**=反向带止损+100% 仓位（基础策略）；**B**=当前带与 VWAP 组合止损；**C**=B+波动目标
  动态仓位（lev=min(4, 2%/σ_14d)）
- 成本：佣金万1/边 + 滑点 ¥0.001/份/边；整手 100 份；初始资金 ¥1,000,000

## 快速开始

```bash
pip install -r requirements.txt

# 复现基础策略：A/B/C 三变体 g30 + RUN-001 锚点断言（强验收）
python scripts/run_base.py --validate

# 单变体/换网格
python scripts/run_base.py --variant A --grid g15 --validate

# 参数遍历（多线程，示例 VM × lookback）
python scripts/run_sweep.py --variant A --grid g30 --sweep-name demo_vm_L --workers 16
```

## 改参与加参

全部参数集中在 `src/intramom/config.py` 的 `StrategyConfig`（默认值见 `config/base.json`）：

| 分组 | 参数 |
|---|---|
| 数据 | `symbol` `data_dir` `start` `end` |
| 信号 | `lookback`(σ 窗口) `vol_mult`(VM) `grid`/`grid_points`(自定义网格) |
| 变体 | `stop_mode`(opp_band/band_vwap/vwap_only) `sizing`(fixed/vol_target) `sigma_target` `max_leverage` `vol_lookback` |
| 成本 | `commission_rate` `slippage_per_share` `lot_size` |
| 账户 | `initial_capital` |

- **改参**：改 `config/base.json`，或代码中 `cfg.with_overrides(vol_mult=1.5)`。
- **加参**：在 `StrategyConfig` 加带默认值的字段即可（序列化自动兼容）；如需进入结果目录名，
  在 `auto_name()` 中追加；如需影响引擎行为，在 `engine.run_backtest` 取用。
- **遍历**：`StrategyConfig.expand(base, vol_mult=[...], lookback=[...])` 或
  `run_sweep(base, {"vol_mult": [...], "grid": ["g30", "g15"]}, ...)`。

## 输出规范（OU_GRID 风格，每次运行一个目录）

`data/results/<run_name>/`：`config_snapshot.json` `params.json` `metrics.json` `meta.json`
`equity_daily.csv` `equity_drawdown.png` `fills.csv`（每笔成交）`round_trips.csv`（整趟）
`strategy_decisions.csv`（每个决策时点：价/σ/UB/LB/VWAP/仓位/动作）
`account_snapshots.csv` `position_snapshots.csv` `backtest_summary.csv` `report.md`。
遍历结果：`data/results/sweeps/<sweep_name>/summary.csv` + 每组合同名目录。

## 数据（只读引用，不复制）

前序冻结快照 `../Intraday_Momentum_for_ETF/data/processed/`（RUN-001，QA 21/21 PASS）：
分钟 bar 526,825 行 × 2,186 日（2017-01-03~2025-12-31，09:30 竞价 bar + END 标签）、
日线（pre_close 除息调整口径）、000300 指数（对照）、分钟级可交易性。路径可在 config 改指。

## 验证锚点（与前序 RUN-001 逐值一致，容差 1e-9）

| 变体×网格 | exec_rows | end_aum | 关键指标 |
|---|---:|---:|---|
| A g30 | 2,840 | 1,222,781.49153 | Sharpe 0.26888，MDD −23.29%，首笔 2017-01-23 10:01 买入 297,600@3.378 |
| B g30 | 3,410 | 811,179.3446699956 | Sharpe −0.24382 |
| C g30 | 3,410 | 419,148.8050699884 | Sharpe −0.53714 |
| A/B/C g15 | — | 逐值一致 | `run_base.py --grid g15 --validate` 已 PASS |

`python scripts/run_base.py --validate` 内置全部锚点断言（`src/intramom/anchors.py`）。

## 后续研究挂载点（本轮未做）

论文 §4 的机制/稳健性分析（日内形态 T5、星期效应 T6、VM 扫描 F9、RSI 回归 §4.5、
成本敏感性 F10、VWAP-only Q22、多 ETF Q12 等）——均可通过 `run_sweep` + 新增分析脚本挂载；
引擎已支持 `vwap_only` 止损与自定义决策网格（`grid="custom"` + `grid_points`）。
