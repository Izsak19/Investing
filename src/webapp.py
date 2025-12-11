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
    step: int
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    action: str
    reward: float
    portfolio_value: float
    cash: float
    position: float


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
                    "metrics": self._latest_metrics,
                }
            return payload

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
        timestamp: object,
        ohlc: Dict[str, float],
        action: str,
        reward: float,
        portfolio_value: float,
        cash: float,
        position: float,
        success_rate: float,
        total_reward: float,
    ) -> None:
        ts_str = "" if timestamp is None else str(timestamp)
        with self._lock:
            next_id = (self._events[-1].id + 1) if self._events else 0
            event = TradeEvent(
                id=next_id,
                step=step,
                timestamp=ts_str,
                open=float(ohlc.get("open", ohlc.get("close", 0.0))),
                high=float(ohlc.get("high", ohlc.get("close", 0.0))),
                low=float(ohlc.get("low", ohlc.get("close", 0.0))),
                close=float(ohlc.get("close", 0.0)),
                action=action,
                reward=float(reward),
                portfolio_value=float(portfolio_value),
                cash=float(cash),
                position=float(position),
            )
            self._events.append(event)
            self._latest_metrics = {
                "success_rate": float(success_rate),
                "total_reward": float(total_reward),
                "portfolio_value": float(portfolio_value),
                "cash": float(cash),
                "position": float(position),
            }

    @property
    def app(self) -> FastAPI:
        return self._app
