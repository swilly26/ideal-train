#!/usr/bin/env python3
"""Resumable Alpaca market-data downloader for the ScalpSet backtest.

Pulls 1-minute OHLCV bars for a symbol set over a long window and caches them
as one Parquet file per symbol per calendar month::

    <cache-dir>/1m/<SYMBOL>/<YYYY-MM>.parquet
    <cache-dir>/1m/<SYMBOL>/<YYYY-MM>.done      # completion marker
    <cache-dir>/manifest.json                   # what actually landed

Why monthly files:

* **Resumable** — the machine this runs on suspends for long stretches
  (host clock has jumped ~38 h).  A partially-downloaded cache is fine: every
  completed month is written to disk immediately, and a re-run skips months
  that already have a ``.done`` marker.  No work is ever redone.
* **Cheap to refresh** — the current (incomplete) month is the only one that
  has to be re-fetched, and ``--refresh-last`` handles that.

Rate limiting / batching: requests are paced (``--sleep``, default 0.32 s →
~185 req/min, under Alpaca's 200/min free-tier ceiling), retried with
exponential backoff on 429/5xx, and paged with ``page_token`` (10 000 bars
per request).

Usage::

    .venv/bin/python scripts/fetch_scalpset_history.py \
        --symbols NVDA,META,QQQ,TSLA,COIN,AVGO \
        --start 2025-09-16 --end 2026-09-16

Credentials come from ``.env`` in the repo root (``ALPACA_API_KEY`` /
``ALPACA_SECRET_KEY``) — never from the shell, so this script can be run
without exporting live trading keys into the environment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DATA_HOST = "https://data.alpaca.markets"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "history"
DEFAULT_SYMBOLS = ("NVDA", "META", "QQQ", "TSLA", "COIN", "AVGO")


def load_env(path: Path | None = None) -> dict:
    """Parse the repo ``.env`` (no export into this process' shell env)."""
    env: dict[str, str] = {}
    p = path or (ROOT / ".env")
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def month_ranges(start: dt.date, end: dt.date):
    """Yield ``(first_day, last_day)`` calendar-month chunks covering [start, end]."""
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        nxt = dt.date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
        first = max(cur, start)
        last = min(nxt - dt.timedelta(days=1), end)
        if first <= last:
            yield first, last
        cur = nxt


class AlpacaData:
    """Thin, paced REST client for Alpaca's ``/v2/stocks`` bar endpoints."""

    def __init__(self, key: str, secret: str, feed: str = "sip",
                 sleep: float = 0.32, max_retries: int = 5) -> None:
        self.key, self.secret, self.feed = key, secret, feed
        self.sleep = sleep
        self.max_retries = max_retries
        self.requests = 0

    def _get(self, path: str, params: dict) -> dict:
        url = f"{DATA_HOST}{path}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
        })
        delay = 1.0
        for attempt in range(self.max_retries):
            time.sleep(self.sleep)
            try:
                self.requests += 1
                with urllib.request.urlopen(req, timeout=60) as fh:
                    return json.loads(fh.read())
            except urllib.error.HTTPError as exc:
                body = exc.read()[:300].decode(errors="replace")
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    print(f"    [{exc.code}] retry in {delay:.0f}s: {body[:120]}", flush=True)
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
            except Exception as exc:  # noqa: BLE001 — transient network
                if attempt < self.max_retries - 1:
                    print(f"    [net] retry in {delay:.0f}s: {exc}", flush=True)
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError("unreachable")

    def bars_page(self, symbol: str, start: dt.date, end: dt.date,
                  page_token: str | None = None,
                  end_ts: dt.datetime | None = None) -> dict:
        if end_ts is None:
            end_ts = dt.datetime.combine(end + dt.timedelta(days=1),
                                         dt.time(0, 0), tzinfo=dt.timezone.utc)
        params = {
            "timeframe": "1Min",
            "start": f"{start.isoformat()}T00:00:00Z",
            "end": end_ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 10000,
            "adjustment": "raw",   # unadjusted: matches live trading prices
            "feed": self.feed,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        return self._get(f"/v2/stocks/{symbol}/bars", params)

    def fetch_month(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Fetch every 1-minute bar in [start, end].

        The SIP entitlement on this account covers *historical* data only —
        a range that reaches into the last ~15 minutes of live tape is rejected
        with ``403 subscription does not permit querying recent SIP data``.  So
        the requested end is clamped to ``now - 20 min`` and, if the API still
        refuses, stepped back a full day.  The backtest window therefore stops
        ~20 minutes before the wall clock, which is documented in the report.
        """
        now = dt.datetime.now(dt.timezone.utc)
        hist_cutoff = now - dt.timedelta(minutes=20)
        requested_end = dt.datetime.combine(end + dt.timedelta(days=1),
                                            dt.time(0, 0), tzinfo=dt.timezone.utc)
        candidates = [requested_end]
        if requested_end > hist_cutoff:
            candidates = [hist_cutoff,
                          dt.datetime.combine(now.date(), dt.time(0, 0),
                                              tzinfo=dt.timezone.utc)]
        last_error: Exception | None = None
        for end_ts in candidates:
            try:
                return self._fetch_range(symbol, start, end, end_ts)
            except RuntimeError as exc:
                if "403" not in str(exc):
                    raise
                last_error = exc
                print(f"    {symbol}: 403 for end={end_ts} — retrying with an older end",
                      flush=True)
        raise last_error if last_error else RuntimeError("fetch failed")

    def _fetch_range(self, symbol: str, start: dt.date, end: dt.date,
                     end_ts: dt.datetime) -> pd.DataFrame:
        rows: list[dict] = []
        token: str | None = None
        while True:
            payload = self.bars_page(symbol, start, end, token, end_ts=end_ts)
            for b in payload.get("bars") or []:
                rows.append(b)
            token = payload.get("next_page_token")
            if not token:
                break
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trades", "vwap"])
        df = pd.DataFrame(rows)
        df = df.rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low",
                                "c": "close", "v": "volume", "n": "trades", "vw": "vwap"})
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[~df.index.duplicated(keep="first")]
        df = df.dropna(subset=["open", "high", "low", "close"])
        return df


def month_path(cache: Path, symbol: str, first: dt.date) -> Path:
    return cache / "1m" / symbol.upper() / f"{first.year:04d}-{first.month:02d}.parquet"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--start", default="2025-09-16")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--feed", default=os.environ.get("ALPACA_DATA_FEED", "sip"))
    ap.add_argument("--sleep", type=float, default=0.32)
    ap.add_argument("--refresh-last", action="store_true",
                    help="re-download the newest month even if already cached")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    env = load_env()
    cache = Path(args.cache_dir)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    months = list(month_ranges(start, end))
    print(f"cache={cache} symbols={symbols} months={len(months)} "
          f"({months[0][0]} .. {months[-1][1]}) feed={args.feed}", flush=True)
    if args.dry_run:
        return 0

    key = env.get("ALPACA_API_KEY") or ""
    secret = env.get("ALPACA_SECRET_KEY") or ""
    if not key or not secret:
        print("FATAL: ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env", file=sys.stderr)
        return 2
    client = AlpacaData(key, secret, feed=args.feed, sleep=args.sleep)

    manifest: dict = {"feed": args.feed, "symbols": {}, "generated": None}
    for sym in symbols:
        manifest["symbols"][sym] = {"months": {}, "bars": 0}
        for i, (first, last) in enumerate(months, 1):
            path = month_path(cache, sym, first)
            marker = path.with_suffix(".done")
            is_latest = (first, last) == months[-1]
            if path.exists() and marker.exists() and not (args.refresh_last and is_latest):
                df = pd.read_parquet(path)
                manifest["symbols"][sym]["months"][first.isoformat()] = len(df)
                manifest["symbols"][sym]["bars"] += len(df)
                print(f"  {sym} {first:%Y-%m}: cached ({len(df)} bars)", flush=True)
                continue
            try:
                df = client.fetch_month(sym, first, last)
            except Exception as exc:  # noqa: BLE001 — keep the rest of the run alive
                print(f"  {sym} {first:%Y-%m}: FAILED — {exc}", flush=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if df.empty:
                # Month with no data (future month / holiday-only): record it so
                # a re-run doesn't hammer the API for it again.
                marker.write_text("empty\n")
                print(f"  {sym} {first:%Y-%m}: no bars (marked)", flush=True)
                continue
            df.to_parquet(path, compression="zstd")
            marker.write_text(f"{len(df)} bars {df.index.min()} .. {df.index.max()}\n")
            manifest["symbols"][sym]["months"][first.isoformat()] = len(df)
            manifest["symbols"][sym]["bars"] += len(df)
            print(f"  {sym} {first:%Y-%m}: {len(df)} bars "
                  f"{df.index.min():%Y-%m-%d %H:%M} .. {df.index.max():%Y-%m-%d %H:%M}",
                  flush=True)
    manifest["generated"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["requests"] = client.requests
    cache.mkdir(parents=True, exist_ok=True)
    mpath = cache / "manifest.json"
    if mpath.exists():  # merge with a previous manifest so resume keeps history
        try:
            old = json.loads(mpath.read_text())
            for sym, meta in old.get("symbols", {}).items():
                if sym not in manifest["symbols"]:
                    manifest["symbols"][sym] = meta
        except Exception:  # noqa: BLE001
            pass
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"done: {client.requests} API requests, manifest at {mpath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
