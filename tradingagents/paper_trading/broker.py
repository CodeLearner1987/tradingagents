"""Minimal client for Alpaca's documented paper-trading and market-data REST APIs.

Safety properties, each covered by tests:

- The trading base URL is fixed to ``https://paper-api.alpaca.markets``; any
  other value raises. There is no code path to the live host.
- Credentials come only from ``ALPACA_PAPER_API_KEY_ID`` /
  ``ALPACA_PAPER_API_SECRET_KEY``. Alpaca paper key IDs start with ``PK``;
  anything else is refused, as is an account whose number is not a paper
  (``PA...``) account.
- Keys travel only in request headers. They never appear in ``repr``, error
  messages, or the event log.
- An order submission that times out or gets a 5xx is reported as
  :class:`UncertainOrderState`, never as success or failure, so the caller
  must look the order up by its ``client_order_id`` before doing anything else.

Endpoints used (Alpaca Trading API v2 / Market Data API v2):
``GET /v2/account``, ``GET /v2/clock``, ``GET /v2/positions``,
``GET /v2/orders``, ``GET /v2/orders/{id}``,
``GET /v2/orders:by_client_order_id``, ``POST /v2/orders``,
``DELETE /v2/orders/{id}``, ``GET /v2/stocks/bars``.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

from tradingagents.paper_trading.config import (
    KEY_ID_ENV,
    MARKET_DATA_URL,
    PAPER_TRADING_URL,
    SECRET_KEY_ENV,
)

PAPER_KEY_PREFIX = "PK"
PAPER_ACCOUNT_PREFIX = "PA"


class BrokerError(RuntimeError):
    """A broker call failed in a way that is known not to have changed anything."""


class OrderRejected(BrokerError):
    """Alpaca answered an order request with a 4xx: the order does not exist."""


class UncertainOrderState(BrokerError):
    """An order request may or may not have reached Alpaca (timeout, 5xx, network)."""


class MissingCredentials(BrokerError):
    pass


def mask(value: str | None, keep: int = 4) -> str:
    if not value:
        return "<unset>"
    return f"{'*' * max(0, len(value) - keep)}{value[-keep:]}"


class AlpacaPaperClient:
    def __init__(
        self,
        key_id: str,
        secret_key: str,
        *,
        base_url: str = PAPER_TRADING_URL,
        data_url: str = MARKET_DATA_URL,
        data_feed: str = "sip",
        session: Any = None,
        timeout: float = 15.0,
    ):
        if base_url.rstrip("/") != PAPER_TRADING_URL:
            raise ValueError(
                f"refusing trading endpoint {base_url!r}: only {PAPER_TRADING_URL} is allowed"
            )
        if data_url.rstrip("/") != MARKET_DATA_URL:
            raise ValueError(f"refusing market-data endpoint {data_url!r}")
        if not key_id or not secret_key:
            raise MissingCredentials(f"set {KEY_ID_ENV} and {SECRET_KEY_ENV}")
        if not key_id.startswith(PAPER_KEY_PREFIX):
            raise ValueError(
                f"{KEY_ID_ENV} does not look like an Alpaca paper key "
                f"(paper key IDs start with {PAPER_KEY_PREFIX!r}); refusing to continue"
            )
        self._base = PAPER_TRADING_URL
        self._data = MARKET_DATA_URL
        self._feed = data_feed
        self._timeout = timeout
        self._session = session or requests.Session()
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}
        self._key_hint = mask(key_id)
        self._secrets = (key_id, secret_key)

    @classmethod
    def from_env(cls, **kwargs) -> AlpacaPaperClient:
        return cls(os.getenv(KEY_ID_ENV, ""), os.getenv(SECRET_KEY_ENV, ""), **kwargs)

    def __repr__(self) -> str:
        return f"AlpacaPaperClient(base={self._base!r}, key={self._key_hint})"

    @property
    def secrets(self) -> tuple[str, str]:
        """For log scrubbing only."""
        return self._secrets

    # --- transport -----------------------------------------------------------

    def _call(self, method: str, url: str, *, order_write: bool = False, **kwargs):
        try:
            resp = self._session.request(
                method, url, headers=self._headers, timeout=self._timeout, **kwargs
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            msg = f"{method} {url.split('?')[0]}: {type(exc).__name__}"
            if order_write:
                raise UncertainOrderState(msg) from None
            raise BrokerError(msg) from None
        if resp.status_code == 404:
            return None
        if resp.status_code >= 500 and order_write:
            raise UncertainOrderState(f"{method} {url}: HTTP {resp.status_code}")
        if resp.status_code >= 400:
            detail = _error_message(resp)
            exc_type = OrderRejected if order_write else BrokerError
            raise exc_type(f"{method} {url.split('?')[0]}: HTTP {resp.status_code}: {detail}")
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    # --- account / market state ----------------------------------------------

    def get_account(self) -> dict:
        account = self._call("GET", f"{self._base}/v2/account")
        if not account:
            raise BrokerError("account endpoint returned nothing")
        number = str(account.get("account_number", ""))
        if not number.startswith(PAPER_ACCOUNT_PREFIX):
            raise BrokerError(
                "account is not an Alpaca paper account (number does not start with "
                f"{PAPER_ACCOUNT_PREFIX!r}); refusing to continue"
            )
        return account

    def get_clock(self) -> dict:
        return self._call("GET", f"{self._base}/v2/clock")

    def list_positions(self) -> list[dict]:
        return self._call("GET", f"{self._base}/v2/positions") or []

    def get_position(self, symbol: str) -> dict | None:
        return self._call("GET", f"{self._base}/v2/positions/{symbol}")

    # --- orders --------------------------------------------------------------

    def list_open_orders(self, symbol: str) -> list[dict]:
        params = {"status": "open", "symbols": symbol, "limit": 500}
        return self._call("GET", f"{self._base}/v2/orders", params=params) or []

    def get_order(self, order_id: str) -> dict | None:
        return self._call("GET", f"{self._base}/v2/orders/{order_id}")

    def get_order_by_client_id(self, client_order_id: str) -> dict | None:
        return self._call(
            "GET",
            f"{self._base}/v2/orders:by_client_order_id",
            params={"client_order_id": client_order_id},
        )

    def submit_order(self, **order) -> dict:
        return self._call("POST", f"{self._base}/v2/orders", json=order, order_write=True)

    def cancel_order(self, order_id: str) -> None:
        """Request cancellation. A 422 (already filled/cancelled) is not an error here;
        callers re-read the order to learn its final state."""
        try:
            self._call("DELETE", f"{self._base}/v2/orders/{order_id}", order_write=True)
        except OrderRejected as exc:
            if "HTTP 422" not in str(exc):
                raise

    # --- market data ---------------------------------------------------------

    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Split/dividend-adjusted daily bars between ``start`` and ``end`` (dates, inclusive)."""
        params = {
            "symbols": symbol, "timeframe": "1Day", "start": start, "end": end,
            "adjustment": "all", "feed": self._feed, "limit": 10000,
        }
        rows: list[dict] = []
        while True:
            page = self._call("GET", f"{self._data}/v2/stocks/bars", params=params) or {}
            rows.extend((page.get("bars") or {}).get(symbol, []))
            token = page.get("next_page_token")
            if not token:
                break
            params = {**params, "page_token": token}
        if not rows:
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame(rows)
        dates = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
        return pd.DataFrame({
            "Date": dates.dt.tz_localize(None).dt.normalize(),
            "Open": df["o"].astype(float), "High": df["h"].astype(float),
            "Low": df["l"].astype(float), "Close": df["c"].astype(float),
            "Volume": df["v"].astype(float),
        })


def _error_message(resp) -> str:
    try:
        body = resp.json()
        return str(body.get("message") or body)[:300]
    except ValueError:
        return (resp.text or "")[:300]
