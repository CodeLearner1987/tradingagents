"""Strategy settings, the paper-only endpoints, and where local state lives."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

# The only trading endpoint this package will talk to. The live host
# (api.alpaca.markets) is deliberately not configurable anywhere.
PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
MARKET_DATA_URL = "https://data.alpaca.markets"

# Credentials are read from these names only. They are distinct from Alpaca's
# generic APCA_* names so a live key exported for another tool is never
# picked up by accident.
KEY_ID_ENV = "ALPACA_PAPER_API_KEY_ID"
SECRET_KEY_ENV = "ALPACA_PAPER_API_SECRET_KEY"

# Symbols approved for trading. Widening this list is an explicit code change.
APPROVED_SYMBOLS: tuple[str, ...] = ("SPY",)


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters of the SMA-crossover rule. Fixed a priori, never fitted."""

    symbol: str = "SPY"
    fast_window: int = 20
    slow_window: int = 50
    stop_loss_pct: float = 0.02
    risk_per_trade_pct: float = 0.005
    # Entries are marketable limit orders capped this far above the prior
    # close, so a gap up can never make an order cost more than available cash.
    entry_limit_pct: float = 0.03
    max_open_positions: int = 1

    def __post_init__(self):
        if self.symbol not in APPROVED_SYMBOLS:
            raise ValueError(f"{self.symbol!r} is not an approved symbol {APPROVED_SYMBOLS}")
        if not 1 <= self.fast_window < self.slow_window:
            raise ValueError("need 1 <= fast_window < slow_window")
        if not 0 < self.stop_loss_pct < 0.5:
            raise ValueError("stop_loss_pct must be in (0, 0.5)")
        if not 0 < self.risk_per_trade_pct <= 0.02:
            raise ValueError("risk_per_trade_pct must be in (0, 0.02]")
        if not 0 < self.entry_limit_pct <= 0.10:
            raise ValueError("entry_limit_pct must be in (0, 0.10]")
        if self.max_open_positions != 1:
            raise ValueError("only one open position is approved")

    def fingerprint(self) -> str:
        """Stable hash of the settings; an approval is bound to it."""
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def state_dir() -> Path:
    """Directory for the event log, state, approval and stop files (outside Git)."""
    default = Path.home() / ".tradingagents" / "paper_trading"
    return Path(os.getenv("TRADINGAGENTS_PAPER_DIR", str(default))).expanduser()
