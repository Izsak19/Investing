# Project Environment (Investing_Codex)

> **Goal of this file**: give an MCP-powered assistant enough context to be productive immediately (project purpose, structure, entrypoints, key configs, and common workflows) without re-discovering the repo every session.

## What this repo is
A lightweight, resource-friendly **BTC/USDT** training loop that learns a simple trading policy from OHLCV candles using common technical indicators, with optional terminal + web dashboards. The core loop is online / streaming-friendly and persists state to disk so training can resume across runs.

Primary audience: experimentation / learning / simulation (not real-money trading).

## Quickstart commands
Create & activate venv, install deps:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Smoke test (offline + terminal dashboard):

```bash
python main.py --offline --dashboard --duration 60
```

Run live (requires internet; uses ccxt under the hood):

```bash
python main.py --dashboard --steps 120
```

Run with the Plotly/FastAPI web dashboard (defaults to port 8000):

```bash
python main.py --offline --web-dashboard --dashboard --duration 120
```

Auto-retrain loop on last 24h window (train → validate → backtest; repeats until threshold met or max cycles):

```bash
python auto_retrain.py --offline --max-cycles 3
```

## Repository layout (high signal)

```text
.
├─ main.py                # primary CLI entrypoint
├─ auto_retrain.py        # repeated train/validate/backtest loop
├─ requirements.txt       # runtime dependencies
├─ README.md              # human docs + usage examples
├─ src/
│  ├─ config.py            # all knobs (fees, gating, exploration, reward shaping, etc.)
│  ├─ data_feed.py         # candle source (live via ccxt or synthetic offline)
│  ├─ indicators.py        # feature engineering (MA/EMA/WMA/BOLL/VWAP/ATR/TRIX/SAR/SuperTrend)
│  ├─ agent.py             # bandit policy + persistent agent state
│  ├─ trainer.py           # portfolio sim + reward computation + training step loop
│  ├─ dashboard.py         # Rich terminal dashboard
│  ├─ webapp.py            # FastAPI/Plotly streaming dashboard
│  ├─ persistence.py       # atomic writes for json/text/csv artifacts
│  └─ templates/index.html # web dashboard UI
└─ data/
   ├─ state.json           # latest agent state pointer
   ├─ runs/                # run artifacts (metrics, trades, checkpoints, etc.)
   └─ cache/               # optional cached datasets
```

## How the system works (mental model)

1. **Data ingestion**: `DataFeed` returns OHLCV frames; either synthetic offline or fetched from an exchange.
2. **Feature engineering**: `compute_indicators()` adds indicator columns to the frame and drops initial NaNs.
3. **Decision**: the agent proposes an action (`sell`, `hold`, `buy`) from the current feature vector.
4. **Execution simulation**: `Trainer` updates a portfolio (cash + position), applying costs and risk/gating rules.
5. **Learning update**: the agent updates its internal parameters online and persists state periodically.
6. **Logging + dashboards**: trades/metrics are written to disk and optionally streamed to terminal/web dashboards.

## Key entrypoints

### `main.py`
Single-run CLI that supports:
- offline or live mode
- bounded runs by `--steps` or `--duration`
- warmup-gated continuous streaming (`--warmup-hours`, `--warmup-profit-target`, `--continuous`)
- `--dashboard` (terminal) and `--web-dashboard` (FastAPI/Plotly)
- checkpointing + periodic flush of trades/metrics into `data/runs/<run_id>/...`

### `auto_retrain.py`
Repeated cycle that:
- fetches last ~24h of candles
- splits into train/validation
- trains a candidate policy
- evaluates on validation and full backtest
- only keeps the new `data/state.json` if it clears profitability / risk thresholds; otherwise restores the previous state

## Persistent artifacts you’ll see

### Global “latest” state
- `data/state.json` — the state pointer updated each run; delete to reset learning.

### Per-run directory
Under `data/runs/<run_id>/`:
- `agent_state_latest.json` + optional `agent_state_step_*.json` checkpoints
- `trainer_state_latest.json` + optional `trainer_state_step_*.json` checkpoints
- `trades.csv`
- `metrics.json`

Tip: for reproducibility, run with `--cache --cache-dir data/cache` and keep the run directory.

## Configuration: where to look first
All important knobs live in `src/config.py`. Highlights:
- **Market defaults**: `DEFAULT_SYMBOL`, `DEFAULT_TIMEFRAME`, `DEFAULT_LIMIT`
- **Costs**: `FEE_RATE`, `TURNOVER_PENALTY`, `SLIPPAGE_RATE`
- **Gating to reduce churn** (critical on 1m): `COST_AWARE_GATING`, `MIN_TRADE_GAP_STEPS`, `MIN_HOLD_STEPS`, `EDGE_THRESHOLD` (+ adaptive/regime gates)
- **Exploration**: `POSTERIOR_SCALE` (+ decay half-life and minimum)
- **Reward shaping / normalization**: `REWARD_SCALE`, `USE_ADVANTAGE_BASELINE`, `USE_REWARD_STD_NORMALIZATION`, tanh clips
- **Risk controls**: drawdown + turnover budgets and penalties
- **Profiles**: `apply_profile(name)` can quickly swap a preset (e.g., more trades vs conservative).

## Indicators / features
The baseline indicator set is defined in `src/indicators.py` as `INDICATOR_COLUMNS` and produced by `compute_indicators(df)`.
Typical columns include MA/EMA/WMA, Bollinger bands, VWAP, ATR, TRIX, SAR, SuperTrend.

## Agent basics (what to know when changing it)
The agent persists a JSON-serializable state (weights, covariances, reward totals, step counts) and can migrate older state to newer feature sets. Keep in mind:
- Feature ordering matters (it’s persisted).
- Numeric safety: features/weights are clipped in several places.
- Exploration is controlled via config and (optionally) CLI overrides in `main.py`.

## Common workflows

### Reset learning
```bash
rm -f data/state.json
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
- Increase `EDGE_THRESHOLD` or `COST_EDGE_MULT`
- Increase `MIN_TRADE_GAP_STEPS` / `MIN_HOLD_STEPS`
- Increase `TURNOVER_BUDGET_PENALTY`

### Make the agent explore more (debugging)
- Increase `POSTERIOR_SCALE`
- Apply the `debug_more_trades` profile
- Reduce gating thresholds temporarily

## Conventions for MCP-assisted changes
When asking an assistant to modify this repo, it helps to specify:
- whether you’re targeting **offline** or **live** behavior
- which artifact should be considered the “source of truth” (global `data/state.json` vs a specific `data/runs/<run_id>/...`)
- whether you want to keep backwards compatibility with existing saved states

## Notes / caveats
- This is a simulation/learning sandbox; it is not a production trading system.
- 1m data is extremely fee-sensitive, so churn control (gating/hysteresis) is a first-order concern.
