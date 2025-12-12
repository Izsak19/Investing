# Investing lightweight BTC/USDT trainer

This project provides a minimal, resource-friendly loop for training a simple trading bandit on 1-minute BTC/USDT candles using common technical indicators (MA, EMA, WMA, Bollinger Bands, VWAP, ATR, TRIX, SAR, SuperTrend). It includes a terminal dashboard to visualize live progress.

## Environment setup (PyCharm-friendly)
1. **Clone & open:** Open the project folder in PyCharm (File → Open…).
2. **Create a virtualenv:**
   - PyCharm: *Python Interpreter* → *Add New Interpreter* → *Virtualenv*, point to this folder.
   - Or manually: `python -m venv .venv` and activate it (`source .venv/bin/activate` or `.venv\\Scripts\\activate`).
3. **Install dependencies:** `pip install -r requirements.txt` (pure Python deps: pandas/numpy/ccxt/rich; no compiled TA libs).
4. **Configure run target:** In PyCharm, create a *Run Configuration* that calls `main.py` with flags you need, e.g. `--offline --dashboard --steps 50`.

## Quick start
```bash
python main.py --offline --dashboard --duration 120
```
- `--duration` runs the loop for a number of **seconds** so you can watch the bandit adapt in real time.
- `--delay` controls the pause between events (default 1s) to keep the dashboard readable.
- `--offline` uses synthetic candles and works even if `ccxt` is not installed.
- Remove `--offline` to fetch live 1m candles via `ccxt` (internet required).

## Online learning cycles (live candles)
The trainer consumes 1m candles by default, so each hour of learning is `60` steps (`60 / 1`). The dashboard (`--dashboard`) refreshes after **every trading action** so you can watch each decision in real time.

- **Custom duration:**
  ```bash
  HOURS=1
  python main.py --dashboard --steps $((HOURS * 60))
  ```

- **2h cycle:**
  ```bash
  python main.py --dashboard --steps 120
  ```

- **6h cycle:**
  ```bash
  python main.py --dashboard --steps 360
  ```

- **12h cycle:**
  ```bash
  python main.py --dashboard --steps 720
  ```

If you prefer a different timeframe (e.g., `5m`), adjust `--timeframe` and rescale the `--steps` count accordingly.

## Profit-gated warmup → continuous streaming
Let the agent prove itself on the last 24h of candles before letting it run unattended:

```bash
# Warm up for 3 hours on the last 24h window until breakeven, then start live streaming
python main.py --dashboard --warmup-hours 3 --warmup-profit-target 0 --continuous

# Demand a 0.5% gain before going live (still capped at the warmup hours)
python main.py --dashboard --warmup-hours 2 --warmup-profit-target 0.5 --continuous
```

- The warmup replays the most recent 24h window on a loop, applying indicators once and pausing per `--delay` so you can
  watch the dashboard evolve.
- Continuous streaming will *only* begin after the portfolio value clears the target on the warmup run; otherwise the run
  stops to avoid unleashing an unprofitable policy.

## Learning loop (modern lightweight cycle)
The code implements a lean version of the typical ML lifecycle:
1. **Data ingestion** — `DataFeed` pulls live candles or generates synthetic data.
2. **Feature engineering** — `compute_indicators` adds MA/EMA/WMA/BOLL/VWAP/ATR/TRIX/SAR/SuperTrend columns.
3. **Policy/action selection** — `BanditAgent` picks *sell/hold/buy* via Thompson-sampled linear rewards that naturally balance exploration and exploitation.
4. **Execution & reward** — `Trainer` simulates trades, computes reward as realized/unrealized PnL.
5. **Learning update** — Online Sherman–Morrison updates keep the linear posterior fresh without full matrix inversions.
6. **Logging/persistence** — agent state saved to `data/state.json`, trades to `data/trades.csv`.
7. **Monitoring** — `--dashboard` renders a Rich-powered terminal dashboard showing the current step, price, action, reward, PnL, and Q-values.

## Knowledge/data management
- **State:** `data/state.json` (Q-values, cumulative reward, trade count) is updated after each run; delete it to reset learning.
- **Trades:** `data/trades.csv` logs every simulated action for lightweight auditability.
- **Minimal footprint:** All data lives in `data/` to keep the flow simple and PyCharm-friendly.

## Suggested model/strategy
- The default **uncertainty-aware contextual bandit** uses Thompson sampling over a linear posterior to favor high-reward contexts while still probing unfamiliar regions without a global epsilon.
- You can experiment with contextual features (e.g., normalized indicators) to condition actions, or swap in a small policy-gradient model if you need more expressiveness.

## Step-by-step model training strategy
Follow this command-first recipe to train and iterate quickly.

1. **Set up your environment** (once per machine)
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. **Smoke test offline loop** (fast functional check, no network)
   ```bash
   python main.py --offline --dashboard --duration 60
   ```
   - Confirms indicators, dashboard rendering, and logging to `data/` work end-to-end.

3. **Warm up on the last 24h window** (profit-gated)  
   Skip this if you only want synthetic data.
   ```bash
   python main.py --dashboard --warmup-hours 2 --warmup-profit-target 0.25 --continuous
   ```
   - Replays the last 24h until it reaches +0.25% profit or the warmup hours elapse, then rolls into live streaming.

4. **Short live training cycle** (1–2h)
   Great for tuning hyperparameters such as `--posterior-scale` (exploration temperature) and `--delay` (UI cadence).
   ```bash
   python main.py --dashboard --steps 120 --posterior-scale 0.35 --delay 1
   ```

5. **Extended live training** (half/whole day)  
   Persist the agent state automatically in `data/state.json` so subsequent runs pick up where you left off.
   ```bash
   python main.py --dashboard --steps 720 --delay 1 --timeframe 1m
   ```

6. **Inspect learning artifacts**
   ```bash
   ls data/
   head -n 20 data/trades.csv
   cat data/state.json | python -m json.tool
   ```
   - Review trades for edge cases and verify Q-values are evolving sensibly.

7. **Automated nightly retrain/backtest**  
   Stops early if validation accuracy misses the target and restores the previous state.
   ```bash
   python auto_retrain.py --min-success-rate 60 --train-steps 500 --validate-steps 120
   ```

8. **Reset & rerun experiments**  
   When you want a clean slate:
   ```bash
   rm -f data/state.json data/trades.csv
   ```
   - Then restart from step 2 with tweaked flags.

## Dashboard
Enable `--dashboard` to see a live table that refreshes as the agent trains. It is text-only to minimize resource usage and works well inside PyCharm's Run window.

## HTML dashboard
For a richer visualization, launch the Plotly-powered web dashboard:

```bash
python main.py --offline --web-dashboard --dashboard --duration 120
```

- Opens a FastAPI server on `http://localhost:8000` (override with `--web-port`).
- Streams OHLC candles, actions, and portfolio metrics to the browser while still rendering the terminal dashboard.
- The console prints a friendly link (`Dashboard available at http://localhost:8000`) even when the server binds to `0.0.0.0`.

## Automated 24h retraining loop
Run a full “fetch → feature → train → validate → backtest” cycle on the last 24h of candles until a target success rate is reached:

```bash
python auto_retrain.py --offline --min-success-rate 60 --max-cycles 3
```

- Uses live exchange candles by default; add `--offline` for synthetic data.
- Splits the window into train/validation slices (default 80/20) and restores the prior agent state when the validation success rate misses the threshold.
- Adjust `--timeframe` or `--train-steps` if you want fewer than 1440 steps for a 1m window.

## Safety & next steps
- This code is for experimentation only—do **not** use it for real-money trading without substantial risk controls.
- For production use, add proper exchange credentials management, slippage/fee modeling, and unit tests for indicator correctness.
