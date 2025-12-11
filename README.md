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

## Learning loop (modern lightweight cycle)
The code implements a lean version of the typical ML lifecycle:
1. **Data ingestion** — `DataFeed` pulls live candles or generates synthetic data.
2. **Feature engineering** — `compute_indicators` adds MA/EMA/WMA/BOLL/VWAP/ATR/TRIX/SAR/SuperTrend columns.
3. **Policy/action selection** — `BanditAgent` picks *sell/hold/buy* via epsilon-greedy Q-values.
4. **Execution & reward** — `Trainer` simulates trades, computes reward as realized/unrealized PnL.
5. **Learning update** — Q-values updated with a simple bandit rule (reward-only, no discount factor).
6. **Logging/persistence** — agent state saved to `data/state.json`, trades to `data/trades.csv`.
7. **Monitoring** — `--dashboard` renders a Rich-powered terminal dashboard showing the current step, price, action, reward, PnL, and Q-values.

## Knowledge/data management
- **State:** `data/state.json` (Q-values, cumulative reward, trade count) is updated after each run; delete it to reset learning.
- **Trades:** `data/trades.csv` logs every simulated action for lightweight auditability.
- **Minimal footprint:** All data lives in `data/` to keep the flow simple and PyCharm-friendly.

## Suggested model/strategy
- The default **epsilon-greedy multi-armed bandit** is intentionally simple and stable for small datasets and short horizons.
- You can experiment with contextual features (e.g., normalized indicators) to condition actions, or swap in a small policy-gradient model if you need more expressiveness.

## Dashboard
Enable `--dashboard` to see a live table that refreshes as the agent trains. It is text-only to minimize resource usage and works well inside PyCharm's Run window.

## Safety & next steps
- This code is for experimentation only—do **not** use it for real-money trading without substantial risk controls.
- For production use, add proper exchange credentials management, slippage/fee modeling, and unit tests for indicator correctness.
