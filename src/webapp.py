from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn


@dataclass
class TradeEvent:
    id: int
    cycle_id: int
    step: int
    episode_step: int
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    action: str
    reward: float
    trainer_reward: float
    mtm_delta: float
    trade_impact: float
    fee_paid: float
    turnover_penalty: float
    trade_size: float
    notional_traded: float
    refilled: bool
    refill_count: int
    success_rate: float
    step_win_rate: float
    total_reward: float
    portfolio_value: float
    cash: float
    position: float
    executed_trades: int
    sell_win_rate: float
    realized_pnl: float
    data_is_live: bool
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    action_hold: float
    action_buy: float
    action_sell: float


class WebDashboard:
    """Lightweight FastAPI server to stream trade events to a browser UI."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000, history: int = 500):
        self.host = host
        self.port = port
        self.history = history

        self._events: Deque[TradeEvent] = deque(maxlen=history)
        self._lock = threading.Lock()
        self._app = FastAPI(title="Trainer Dashboard")
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

        self._latest_metrics: Dict[str, float] = {}
        self._cycle_id: int = 0
        self._next_event_id: int = 0
        self._load_routes()

    def _load_routes(self) -> None:
        templates_dir = Path(__file__).resolve().parent / "templates"
        index_path = templates_dir / "index.html"
        index_html = index_path.read_text() if index_path.exists() else "<h1>Dashboard missing</h1>"

        @self._app.get("/", response_class=HTMLResponse)
        def index() -> str:  # pragma: no cover - exercised at runtime
            return index_html

        @self._app.get("/api/events")
        def events(since: int = -1) -> Dict[str, object]:  # pragma: no cover - exercised at runtime
            with self._lock:
                filtered = [asdict(ev) for ev in self._events if ev.id > since]
                last_id = self._events[-1].id if self._events else since
                payload = {
                    "events": filtered,
                    "last_id": last_id,
                    "cycle_id": self._cycle_id,
                    "metrics": self._latest_metrics,
                }
            return payload

    def reset_cycle(self) -> None:
        """Clear the event buffer and advance the cycle counter.

        This is used when the trainer starts a new episode/window so the browser
        chart doesn't accumulate overlapping candles and markers from prior cycles.
        """
        with self._lock:
            self._events.clear()
            self._latest_metrics = {}
            self._cycle_id += 1

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        config = uvicorn.Config(self._app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        display_host = "localhost" if self.host in {"0.0.0.0", "::"} else self.host
        print(f"[web] Dashboard available at http://{display_host}:{self.port}")

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=2)

    def publish_event(
        self,
        step: int,
        *,
        episode_step: int = 0,
        timestamp: object | None = None,
        ohlc: Dict[str, float] | None = None,
        action: str,
        reward: float,
        portfolio_value: float,
        cash: float,
        position: float,
        success_rate: float,
        step_win_rate: float,
        total_reward: float,
        trainer_reward: float,
        mtm_delta: float,
        trade_impact: float,
        fee_paid: float,
        turnover_penalty: float,
        trade_size: float,
        notional_traded: float,
        refilled: bool,
        refill_count: int,
        executed_trades: int,
        sell_win_rate: float,
        realized_pnl: float,
        data_is_live: bool,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        action_distribution: Dict[str, float] | None = None,
    ) -> None:
        ts_str = "" if timestamp is None else str(timestamp)
        action_distribution = action_distribution or {}
        ohlc = ohlc or {}
        with self._lock:
            next_id = self._next_event_id
            self._next_event_id += 1
            event = TradeEvent(
                id=next_id,
                cycle_id=int(self._cycle_id),
                step=int(step),
                episode_step=int(episode_step),
                timestamp=ts_str,
                open=float(ohlc.get("open", ohlc.get("close", 0.0))),
                high=float(ohlc.get("high", ohlc.get("close", 0.0))),
                low=float(ohlc.get("low", ohlc.get("close", 0.0))),
                close=float(ohlc.get("close", 0.0)),
                action=action,
                reward=float(reward),
                trainer_reward=float(trainer_reward),
                mtm_delta=float(mtm_delta),
                trade_impact=float(trade_impact),
                fee_paid=float(fee_paid),
                turnover_penalty=float(turnover_penalty),
                trade_size=float(trade_size),
                notional_traded=float(notional_traded),
                refilled=bool(refilled),
                refill_count=int(refill_count),
                success_rate=float(success_rate),
                step_win_rate=float(step_win_rate),
                total_reward=float(total_reward),
                portfolio_value=float(portfolio_value),
                cash=float(cash),
                position=float(position),
                executed_trades=int(executed_trades),
                sell_win_rate=float(sell_win_rate),
                realized_pnl=float(realized_pnl),
                data_is_live=bool(data_is_live),
                total_return=float(total_return),
                sharpe_ratio=float(sharpe_ratio),
                max_drawdown=float(max_drawdown),
                action_hold=float(action_distribution.get("hold", 0.0)),
                action_buy=float(action_distribution.get("buy", 0.0)),
                action_sell=float(action_distribution.get("sell", 0.0)),
            )
            self._events.append(event)
            self._latest_metrics = {
                "action": action,
                "success_rate": float(success_rate),
                "step_win_rate": float(step_win_rate),
                "total_reward": float(total_reward),
                "portfolio_value": float(portfolio_value),
                "cash": float(cash),
                "position": float(position),
                "trainer_reward": float(trainer_reward),
                "mtm_delta": float(mtm_delta),
                "trade_impact": float(trade_impact),
                "fee_paid": float(fee_paid),
                "turnover_penalty": float(turnover_penalty),
                "trade_size": float(trade_size),
                "notional_traded": float(notional_traded),
                "refilled": bool(refilled),
                "refill_count": int(refill_count),
                "executed_trades": int(executed_trades),
                "sell_win_rate": float(sell_win_rate),
                "realized_pnl": float(realized_pnl),
                "data_is_live": bool(data_is_live),
                "total_return": float(total_return),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "action_hold": float(action_distribution.get("hold", 0.0)),
                "action_buy": float(action_distribution.get("buy", 0.0)),
                "action_sell": float(action_distribution.get("sell", 0.0)),
            }

    @property
    def app(self) -> FastAPI:
        return self._app
