"""Alpaca paper client safety (tradingagents.paper_trading.broker). No network."""

import pytest
import requests

from tradingagents.paper_trading.broker import (
    AlpacaPaperClient,
    BrokerError,
    MissingCredentials,
    OrderRejected,
    UncertainOrderState,
)

KEY, SECRET = "PKTEST1234567890", "s3cr3t-value-never-logged"


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.content = b"x" if payload is not None or text else b""

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, headers=None, timeout=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": headers, **kwargs})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def client(*responses):
    session = FakeSession(*responses)
    return AlpacaPaperClient(KEY, SECRET, session=session), session


@pytest.mark.unit
@pytest.mark.parametrize("url", ["https://api.alpaca.markets", "http://paper-api.alpaca.markets",
                                 "https://paper-api.alpaca.markets.evil.com"])
def test_refuses_any_non_paper_trading_endpoint(url):
    with pytest.raises(ValueError, match="only https://paper-api.alpaca.markets"):
        AlpacaPaperClient(KEY, SECRET, base_url=url)


@pytest.mark.unit
def test_refuses_non_paper_key_and_missing_credentials(monkeypatch):
    with pytest.raises(ValueError, match="paper key"):
        AlpacaPaperClient("AKLIVEKEY123", SECRET)
    monkeypatch.delenv("ALPACA_PAPER_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_API_SECRET_KEY", raising=False)
    with pytest.raises(MissingCredentials):
        AlpacaPaperClient.from_env()


@pytest.mark.unit
def test_generic_apca_env_vars_are_ignored(monkeypatch):
    monkeypatch.delenv("ALPACA_PAPER_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_API_SECRET_KEY", raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", SECRET)
    with pytest.raises(MissingCredentials):
        AlpacaPaperClient.from_env()


@pytest.mark.unit
def test_repr_masks_key_and_never_shows_secret():
    c, _ = client()
    text = repr(c)
    assert SECRET not in text and KEY not in text and KEY[-4:] in text


@pytest.mark.unit
def test_refuses_live_account_number():
    c, _ = client(FakeResponse(200, {"account_number": "123456789", "status": "ACTIVE"}))
    with pytest.raises(BrokerError, match="not an Alpaca paper account"):
        c.get_account()


@pytest.mark.unit
def test_requests_go_to_paper_host_with_header_auth():
    c, s = client(FakeResponse(200, {"account_number": "PA1234", "status": "ACTIVE"}))
    assert c.get_account()["account_number"] == "PA1234"
    call = s.calls[0]
    assert call["url"] == "https://paper-api.alpaca.markets/v2/account"
    assert call["headers"] == {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET}


@pytest.mark.unit
def test_order_outcomes_are_classified():
    c, _ = client(
        requests.Timeout("read timed out"),
        FakeResponse(503, {"message": "unavailable"}),
        FakeResponse(403, {"message": "insufficient buying power"}),
        FakeResponse(200, {"id": "o1", "status": "accepted"}),
    )
    with pytest.raises(UncertainOrderState):
        c.submit_order(symbol="SPY", qty="1")
    with pytest.raises(UncertainOrderState):
        c.submit_order(symbol="SPY", qty="1")
    with pytest.raises(OrderRejected, match="insufficient buying power") as exc:
        c.submit_order(symbol="SPY", qty="1")
    assert SECRET not in str(exc.value) and KEY not in str(exc.value)
    assert c.submit_order(symbol="SPY", qty="1")["id"] == "o1"


@pytest.mark.unit
def test_missing_things_are_none_and_cancel_tolerates_422():
    c, _ = client(FakeResponse(404, {"message": "position does not exist"}),
                  FakeResponse(404, {"message": "order not found"}),
                  FakeResponse(422, {"message": "order is not cancelable"}),
                  FakeResponse(204))
    assert c.get_position("SPY") is None
    assert c.get_order_by_client_id("nope") is None
    c.cancel_order("filled-already")
    c.cancel_order("open-order")


@pytest.mark.unit
def test_read_errors_are_broker_errors_not_uncertain():
    c, _ = client(requests.ConnectionError("down"), FakeResponse(500, {"message": "x"}))
    with pytest.raises(BrokerError) as e1:
        c.get_clock()
    assert not isinstance(e1.value, UncertainOrderState)
    with pytest.raises(BrokerError) as e2:
        c.list_positions()
    assert not isinstance(e2.value, UncertainOrderState)


@pytest.mark.unit
def test_daily_bars_paginate_and_use_new_york_dates():
    page1 = {"bars": {"SPY": [{"t": "2024-06-03T04:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10}]},
             "next_page_token": "abc"}
    page2 = {"bars": {"SPY": [{"t": "2024-06-04T04:00:00Z", "o": 2, "h": 3, "l": 1.5, "c": 2.5, "v": 20}]},
             "next_page_token": None}
    c, s = client(FakeResponse(200, page1), FakeResponse(200, page2))
    bars = c.get_daily_bars("SPY", "2024-06-01", "2024-06-05")
    assert bars["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-06-03", "2024-06-04"]
    assert bars["Close"].tolist() == [1.5, 2.5]
    assert s.calls[0]["url"] == "https://data.alpaca.markets/v2/stocks/bars"
    assert s.calls[0]["params"]["adjustment"] == "all"
    assert s.calls[1]["params"]["page_token"] == "abc"
