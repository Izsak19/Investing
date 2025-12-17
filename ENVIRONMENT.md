# Project Environment (Investing_Codex)

> **Goal of this file**: give an MCP-powered assistant enough context to be productive immediately (project purpose, structure, entrypoints, key configs, and common workflows) without re-discovering the repo every session.

## What this repo is
A lightweight, resource-friendly **BTC/USDT** trading-simulation + learning loop that learns an online policy from OHLCV candles (and derived microstructure-style features). It supports:
- offline (synthetic) or live exchange candles (via `ccxt` through `src/data_feed.py`)
- terminal dashboard and an optional web (FastAPI/Plotly) dashboard
- checkpointing + resuming from disk across runs
- an auto-retrain cycle and an evaluator sweep/promoter over checkpoints

Primary audience: experimentation / learning / simulation (**not** real-money trading).

## Quickstart commands
Create & activate venv, install deps:

```bash
python -m venv .venv
# Windows
.\\.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Smoke test (offline + terminal dashboard):

```bash
python main.py --offline --dashboard --duration 60
```

Run live (requires internet):

```bash
python main.py --dashboard --steps 120
```

Run with the FastAPI/Plotly web dashboard (defaults to port 8000):

```bash
python main.py --offline --web-dashboard --dashboard --duration 120
```

Auto-retrain loop on last ~24h window (train → validate → backtest; repeats until threshold met or max cycles):

```bash
python auto_retrain.py --offline --max-cycles 3
```

Evaluate without training/saving state (bounded pass):

```bash
python main.py --offline --eval --steps 1500
```

## Repository layout (high signal)

```text
.
├─ main.py                # primary CLI entrypoint
├─ auto_retrain.py        # repeated train/validate/backtest loop + evaluator sweep
├─ eval_checkpoints.py    # evaluation helper (if present in your workflow)
├─ requirements.txt       # runtime dependencies
├─ README.md              # human docs + usage examples
├─ src/
│  ├─ config.py            # all knobs (fees, gating, exploration, reward shaping, kill-switch, profiles)
│  ├─ data_feed.py         # candle source (live via ccxt or synthetic offline) + caching
│  ├─ indicators.py        # feature engineering (microstructure/flow/regime features)
│  ├─ metrics.py           # drawdown, Sharpe, return helpers
│  ├─ timeframe.py         # timeframe → minutes utility
│  ├─ agent.py             # agents + persistent agent state
│  ├─ trainer.py           # portfolio sim + reward + training step loop + persistence
│  ├─ dashboard.py         # rich terminal dashboard
│  ├─ webapp.py            # FastAPI/Plotly streaming dashboard
│  ├─ persistence.py       # atomic writes for json/text/csv artifacts
│  └─ eval/                # evaluator: sweep checkpoints + promotion policy
│     ├─ latest.py         # writes/reads data/runs/LATEST.json pointer
│     ├─ orchestrator.py   # sweep_checkpoints(), maybe_promote()
│     ├─ promotion.py      # PromotionPolicy + promotion decision logic
│     ├─ evaluator.py      # run evaluator pass
│     ├─ pnl.py / scoring.py / types.py
│     └─ checkpoints.py
└─ data/
   ├─ state.json           # global agent state used by some flows (see note below)
   ├─ runs/                # per-run artifacts (metrics, trades, checkpoints, eval reports)
   └─ cache/               # optional cached datasets
```

## How the system works (mental model)

1. **Data ingestion**: `DataFeed` returns OHLCV frames; either synthetic offline or fetched from an exchange.
2. **Feature engineering**: `compute_indicators()` adds feature columns (see below).
3. **Decision**: the agent proposes an action (`sell`, `hold`, `buy`) from the current feature vector.
4. **Execution simulation**: `Trainer` updates a portfolio (cash + position), applying costs and gating/risk rules.
5. **Learning update**: the agent updates online (optionally with TD + forgetting) and persists state/checkpoints.
6. **Logging + dashboards**: trades/metrics are written to disk and optionally streamed to terminal/web dashboards.

## Key entrypoints

### `main.py`
Single-run CLI that supports:
- offline or live mode
- bounded runs via `--steps` or `--duration` (duration is seconds)
- warmup-gated continuous streaming (`--warmup-hours`, `--warmup-profit-target`, `--continuous`)
- `--dashboard` (terminal) and `--web-dashboard` (FastAPI/Plotly)
- checkpointing + periodic flush of trades/metrics into `data/runs/<run_id>/...`
- `--eval` to run an evaluation pass (no training; no state/checkpoint writes)
- profiles: `--profile <name>` and penalty weighting via `--penalty-profile {train,eval}`

### `auto_retrain.py`
Repeated cycle that:
- fetches last ~24h of candles (or offline synthetic)
- splits into train/validation
- trains a candidate policy
- evaluates on validation and full backtest
- only keeps the updated global state if it clears profitability / risk thresholds
- after each cycle, tries to run an **evaluator sweep** across checkpoints and optionally **promote** a checkpoint

## Persistent artifacts you’ll see

### Global pointers / state
- `data/runs/LATEST.json` — a pointer written by the evaluator utilities (and `auto_retrain.py`) so tooling can refer to the latest run without guessing by timestamps.
- `data/state.json` — a JSON state file used by some workflows (not a “pointer”; it’s the serialized state). Deleting it resets learning for flows that rely on it.

### Per-run directory
Under `data/runs/<run_id>/` you’ll commonly see:
- `agent_state_latest.json` and optional `agent_state_step_*.json` checkpoints
- `trainer_state_latest.json` and optional `trainer_state_step_*.json` checkpoints
- `trades.csv`
- `metrics.json`
- `seeds.json` (offline mode seeds for reproducibility)
- evaluator outputs (when the eval suite runs), e.g. `eval_report.json` and `promoted_checkpoint.json`

Tip: for reproducibility, run with `--cache --cache-dir data/cache` and keep the run directory.

## Configuration: where to look first
All important knobs live in `src/config.py`. Highlights:
- **Market defaults**: `DEFAULT_SYMBOL`, `DEFAULT_TIMEFRAME`, `DEFAULT_LIMIT`
- **Paths**: `STATE_PATH`, `RUNS_DIR`, cache directory flags
- **Costs**: `FEE_RATE`, `SLIPPAGE_RATE`, `MIN_TRADE_NOTIONAL` (execution), `TURNOVER_PENALTY` (learning regularizer)
- **Anti-churn controls**: cooldowns (`MIN_TRADE_GAP_STEPS`, `MIN_HOLD_STEPS`), trade-rate throttle (`MAX_TRADES_PER_WINDOW`, etc.)
- **Cost-aware gating**: `COST_AWARE_GATING`, `EDGE_THRESHOLD`, `COST_EDGE_MULT`, safety margins
- **Exploration**: Thompson sampling via `POSTERIOR_SCALE` (+ decay/min), plus CLI override `--posterior-scale`
- **Forgetting**: `FORGETTING_FACTOR` and CLI `--forgetting-factor`
- **Reward shaping & normalization**: `REWARD_SCALE` (+ adaptive scaling), advantage baseline, std-normalization, tanh clip
- **Risk controls**: drawdown/turnover budgets & penalties
- **Kill-switch**: `ENABLE_KILL_SWITCH` + thresholds to stop runs early when expectancy/drawdown/trade-cap triggers
- **Profiles**: `apply_profile(name)` plus built-ins like `debug_more_trades`, `conservative`, `tf_5m_conservative`

## Indicators / features
Feature columns are defined in `src/indicators.py` as `INDICATOR_COLUMNS` and produced by `compute_indicators(df)`.

Current feature set (high level groups):
- **Core microstructure**: `ret_1m`, `rv_1m`, `ofi_l1`, `imb1`, `micro_bias`, `rel_spread`, `aggr_imb`, `dw_spread`
- **Flow & positioning proxies**: `cvd_1m`, `whale_net_rate_1m`, `liq_net_rate_1m`, `oi_delta_1m`, `basis_pct`, `funding_x_time`
- **Regime & seasonality**: `rv1m_pct_5m`, `spread_pct_5m`, `tod_sin`, `tod_cos`

If you change features:
- update `INDICATOR_COLUMNS`
- maintain stable ordering (it is persisted in agent state)
- ensure safe division/clipping remains intact (`_safe_divide`, `.clip(...)`)

## Agent basics (what to know when changing it)
The main CLI uses `RLSForgettingAgent` (in `src/agent.py`) which persists a JSON-serializable state (weights, covariances, totals, step counts). Keep in mind:
- feature ordering matters (persisted)
- numeric safety: feature values and weights are clipped and NaNs are sanitized
- exploration is controlled by config and CLI overrides in `main.py`

`auto_retrain.py` currently uses `BanditAgent` (also in `src/agent.py`) for its training loop.

## Common workflows

### Reset learning
```bash
rm -f data/state.json
rm -f data/runs/LATEST.json
# (optional) also clear run outputs
rm -rf data/runs/*
```

### Inspect a run
```bash
ls data/runs/<run_id>/
head -n 30 data/runs/<run_id>/trades.csv
python -m json.tool data/runs/<run_id>/metrics.json
```

### Make the agent trade less
- Increase `EDGE_THRESHOLD` and/or `COST_EDGE_MULT`
- Increase `MIN_TRADE_GAP_STEPS` / `MIN_HOLD_STEPS`
- Lower `MAX_TRADES_PER_WINDOW`
- Increase `TURNOVER_BUDGET_PENALTY`

### Make the agent explore more (debugging)
- Increase `POSTERIOR_SCALE` (or use `--posterior-scale`)
- Apply `--profile debug_more_trades`
- Temporarily relax cooldowns and gating thresholds

## Conventions for MCP-assisted changes
When asking an assistant to modify this repo, it helps to specify:
- whether you’re targeting **offline** or **live** behavior
- which artifact should be considered the “source of truth” (`data/state.json` vs a specific `data/runs/<run_id>/...`)
- whether you want to keep backwards compatibility with existing saved states/checkpoints

## Notes / caveats
- This is a simulation/learning sandbox; it is not a production trading system.
- 1m data is extremely fee-sensitive, so churn control (gating/hysteresis/throttles) is a first-order concern.
