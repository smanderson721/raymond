"""
Alpaca intraday Relative-Volume stream.

This is a *daemon* (not a periodic scan): a single long-running coroutine
that maintains a websocket to Alpaca's market-data IEX feed, listens for
minute bars on the top-N watchlist tickers, and emits awards when the
rolling 5-minute volume exceeds the expected baseline by a factor of
``RVOL_AWARD_THRESHOLD``.

Why this lives outside of SCHEDULES
-----------------------------------
The other scans poll on a schedule. RVOL is fundamentally streamed —
opening and closing a websocket every N seconds would burn through
Alpaca's connection budget and miss bars. So the daemon registers this
under ``DAEMONS`` in ``live_daemon.py`` and the scheduler leaves it alone.

Baseline / expected volume
--------------------------
We use yfinance's 20-day average daily volume (consolidated tape) divided
by 78 (390 trading minutes / 5-min bucket) as the expected per-5-min
bucket volume. The actual stream is IEX-only — so realised volume is a
fraction of the consolidated baseline, which would artificially suppress
RVOL. We compensate with ``IEX_TAPE_SHARE`` (a rough average of IEX's
share of total US equity volume). This is a 1-decimal-place
approximation; calibration of per-ticker IEX share is left for later.

Auth and protocol
-----------------
The Alpaca v2 IEX stream:

  wss://stream.data.alpaca.markets/v2/iex

  → {"action":"auth","key":"<KEY>","secret":"<SECRET>"}
  ← [{"T":"success","msg":"authenticated"}]
  → {"action":"subscribe","bars":["AAPL","MSFT",...]}
  ← bar messages like [{"T":"b","S":"AAPL","o":...,"v":12345,"t":"..."}]

Shutdown is cooperative: when ``stop_event`` fires we close the socket
and return; the daemon's main loop awaits the task and moves on.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import aiohttp

from research.live_score_engine import Session, LIVE_DIR


DAEMON_NAME = "rvol_stream"

# Tunables — calibrated to fire ~5-15x per market-hours day on a top-30 list
TOP_N = 25                       # how many top watchlist tickers we subscribe to
WATCHLIST_FILE = Path(LIVE_DIR) / "watchlist.json"
WATCHLIST_REFRESH_SEC = 60       # re-check watchlist + resubscribe diffs

WINDOW_MIN = 5                   # rolling-window length for "current" volume
RVOL_AWARD_THRESHOLD = 3.0       # award when rvol >= this
RVOL_HIT_THRESHOLD = 6.0         # banner-eligible award when rvol >= this
MIN_WINDOW_VOLUME = 50_000       # ignore micro-volume noise
DEDUP_SEC = 5 * 60               # don't award the same ticker more often than this

# IEX is roughly ~2% of US equity volume on average; the actual share
# varies a lot by ticker. We pick a single fudge factor here so the
# expected per-5-min volume isn't 50× too high. Future revision: per-
# ticker IEX share from a calibration pass.
IEX_TAPE_SHARE = 0.025

# Cache the 20-day avg daily volume per ticker on disk; refresh once a day.
BASELINE_CACHE = Path(LIVE_DIR) / "_rvol_baselines.json"
BASELINE_TTL_SEC = 24 * 3600
TRADING_MINUTES_PER_DAY = 390
BUCKETS_PER_DAY = TRADING_MINUTES_PER_DAY // WINDOW_MIN  # 78

# Status sidecar so the dashboard can show daemon health
STATUS_FILE = Path(LIVE_DIR) / "_stream_status.json"

ALPACA_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"
ALPACA_KEY_ENV = "ALPACA_API_KEY_ID"
ALPACA_SEC_ENV = "ALPACA_API_SECRET_KEY"

# Reconnect backoff
RECONNECT_BACKOFF_INIT = 2.0
RECONNECT_BACKOFF_MAX = 60.0


# ── baseline loading ──────────────────────────────────────────────────

def _load_baselines() -> dict[str, float]:
    """Return ticker → 20d avg daily volume, using cache if fresh."""
    if BASELINE_CACHE.exists():
        try:
            blob = json.loads(BASELINE_CACHE.read_text())
            if time.time() - blob.get("updated_ts", 0) < BASELINE_TTL_SEC:
                return {k: float(v) for k, v in blob.get("baselines", {}).items()}
        except Exception:
            pass
    return {}


def _save_baselines(baselines: dict[str, float]) -> None:
    BASELINE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_CACHE.write_text(json.dumps({
        "updated_ts": time.time(),
        "updated_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baselines": baselines,
    }))


def _fetch_baselines_sync(tickers: list[str]) -> dict[str, float]:
    """Fetch 20-day avg daily volume from yfinance for the given tickers.

    Runs in a thread pool. Returns the merged baseline (existing cache
    plus any newly-fetched tickers).
    """
    import yfinance as yf
    current = _load_baselines()
    missing = [t for t in tickers if t not in current]
    if not missing:
        return current

    # yfinance bulk download is fastest; one call for all missing tickers
    try:
        data = yf.download(
            tickers=" ".join(missing),
            period="1mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
    except Exception:
        data = None

    if data is None or data.empty:
        return current

    for t in missing:
        try:
            if len(missing) == 1:
                vols = data["Volume"]
            else:
                vols = data[t]["Volume"]
            vols = vols.dropna()
            if len(vols) >= 5:
                # Trim to last 20 sessions and take the mean
                current[t] = float(vols.tail(20).mean())
        except Exception:
            continue

    _save_baselines(current)
    return current


# ── watchlist plumbing ────────────────────────────────────────────────

def _load_watchlist_topN() -> list[str]:
    """Read the daemon's watchlist.json and return the top-N tickers.

    Tickers are filtered to plain alphabetic symbols + optional ``.X``
    class suffix (e.g. ``BRK.B``). The upstream watchlist occasionally
    contains odd entries like ``FDXF#`` (placeholder rows from broken
    Form 4 cluster rows) — those would break yfinance baseline lookups
    and get rejected by Alpaca anyway.
    """
    if not WATCHLIST_FILE.exists():
        return []
    try:
        blob = json.loads(WATCHLIST_FILE.read_text())
    except Exception:
        return []
    out: list[str] = []
    for r in blob.get("watchlist", []):
        tk = (r.get("ticker") or "").upper()
        # allow A-Z and a single .X class suffix
        if not tk:
            continue
        head, _, tail = tk.partition(".")
        if not head.isalpha() or (tail and not tail.isalpha()):
            continue
        out.append(tk)
        if len(out) >= TOP_N:
            break
    return out


# ── status sidecar ────────────────────────────────────────────────────

def _write_status(state: dict[str, Any]) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        blob = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}
    except Exception:
        blob = {}
    blob[DAEMON_NAME] = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **state,
    }
    blob["_updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(blob, separators=(",", ":")))
    tmp.replace(STATUS_FILE)


# ── per-ticker rolling state ──────────────────────────────────────────

class TickerState:
    """Track the last WINDOW_MIN minutes of bar volumes for one ticker."""
    __slots__ = ("bars", "last_award_ts")

    def __init__(self) -> None:
        # deque of (epoch_minute, volume)
        self.bars: deque[tuple[int, int]] = deque(maxlen=WINDOW_MIN * 4)
        self.last_award_ts: float = 0.0

    def add_bar(self, ts_iso: str, volume: int) -> None:
        try:
            dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        except Exception:
            return
        epoch_min = int(dt.timestamp() // 60)
        # replace if same minute (Alpaca occasionally re-emits a bar)
        if self.bars and self.bars[-1][0] == epoch_min:
            self.bars[-1] = (epoch_min, volume)
        else:
            self.bars.append((epoch_min, volume))

    def window_volume(self, now_epoch_min: int) -> int:
        cutoff = now_epoch_min - WINDOW_MIN
        return sum(v for m, v in self.bars if m > cutoff)


# ── main daemon loop ──────────────────────────────────────────────────

async def run(stop_event: asyncio.Event, bus, log) -> None:
    """Long-running coroutine. Owns the Alpaca websocket connection.

    `bus` is the live_daemon Broadcaster. We publish a `stream` event on
    each award so the dashboard can show the live feed without waiting
    for the file watcher.
    """
    key = os.environ.get(ALPACA_KEY_ENV)
    secret = os.environ.get(ALPACA_SEC_ENV)
    if not key or not secret:
        log.warning("%s: missing %s / %s; daemon will not start",
                    DAEMON_NAME, ALPACA_KEY_ENV, ALPACA_SEC_ENV)
        _write_status({"connected": False, "error": "no_credentials"})
        return

    states: dict[str, TickerState] = {}
    subscribed: set[str] = set()
    baselines: dict[str, float] = _load_baselines()
    backoff = RECONNECT_BACKOFF_INIT

    log.info("%s: starting", DAEMON_NAME)

    while not stop_event.is_set():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    ALPACA_WS_URL,
                    heartbeat=30,
                    autoping=True,
                ) as ws:
                    # ── auth ──
                    await ws.send_json({"action": "auth", "key": key, "secret": secret})
                    auth_ok = False
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        payload = json.loads(msg.data)
                        if isinstance(payload, list):
                            for item in payload:
                                t = item.get("T")
                                if t == "success" and item.get("msg") == "authenticated":
                                    auth_ok = True
                                elif t == "error":
                                    log.error("%s: auth error %s", DAEMON_NAME, item)
                                    raise RuntimeError(f"alpaca auth error: {item}")
                        if auth_ok:
                            break
                    if not auth_ok:
                        raise RuntimeError("auth flow ended without success")

                    log.info("%s: authenticated", DAEMON_NAME)
                    backoff = RECONNECT_BACKOFF_INIT
                    _write_status({
                        "connected": True,
                        "subscribed_count": 0,
                        "last_bar_at": None,
                    })

                    last_resub = 0.0
                    last_bar_at: Optional[str] = None
                    loop = asyncio.get_running_loop()

                    while not stop_event.is_set():
                        # ── periodic resubscribe to current top-N ──
                        if time.time() - last_resub > WATCHLIST_REFRESH_SEC:
                            wanted = set(_load_watchlist_topN())
                            to_add = wanted - subscribed
                            to_drop = subscribed - wanted
                            if to_add:
                                # refresh baselines for new tickers (in pool to avoid blocking)
                                new_base = await loop.run_in_executor(
                                    None, _fetch_baselines_sync, list(to_add)
                                )
                                baselines.update(new_base)
                                await ws.send_json({
                                    "action": "subscribe",
                                    "bars": sorted(to_add),
                                })
                                log.info("%s: +subscribe %d (%s)", DAEMON_NAME,
                                         len(to_add), ",".join(sorted(to_add))[:120])
                            if to_drop:
                                await ws.send_json({
                                    "action": "unsubscribe",
                                    "bars": sorted(to_drop),
                                })
                                log.info("%s: -unsubscribe %d", DAEMON_NAME, len(to_drop))
                            subscribed = wanted
                            last_resub = time.time()
                            _write_status({
                                "connected": True,
                                "subscribed_count": len(subscribed),
                                "last_bar_at": last_bar_at,
                            })

                        # ── read next message (with timeout so we recheck stop) ──
                        try:
                            msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                        except asyncio.TimeoutError:
                            continue

                        if msg.type in (aiohttp.WSMsgType.CLOSED,
                                        aiohttp.WSMsgType.CLOSING,
                                        aiohttp.WSMsgType.CLOSE,
                                        aiohttp.WSMsgType.ERROR):
                            log.warning("%s: ws closed (%s)", DAEMON_NAME, msg.type)
                            break
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue

                        try:
                            payload = json.loads(msg.data)
                        except Exception:
                            continue
                        if not isinstance(payload, list):
                            continue

                        for item in payload:
                            if item.get("T") != "b":
                                continue
                            tk = item.get("S")
                            vol = int(item.get("v", 0))
                            ts = item.get("t")
                            if not tk or vol <= 0 or not ts:
                                continue
                            last_bar_at = ts
                            st = states.setdefault(tk, TickerState())
                            st.add_bar(ts, vol)

                            # compute rvol against the IEX-adjusted baseline
                            daily_avg = baselines.get(tk)
                            if not daily_avg:
                                continue
                            expected_5min = (daily_avg / BUCKETS_PER_DAY) * IEX_TAPE_SHARE
                            if expected_5min <= 0:
                                continue
                            now_epoch_min = int(datetime.fromisoformat(
                                ts.replace("Z", "+00:00")
                            ).timestamp() // 60)
                            wvol = st.window_volume(now_epoch_min)
                            if wvol < MIN_WINDOW_VOLUME:
                                continue
                            rvol = wvol / expected_5min
                            if rvol < RVOL_AWARD_THRESHOLD:
                                continue
                            # dedup per ticker
                            now = time.time()
                            if now - st.last_award_ts < DEDUP_SEC:
                                continue
                            st.last_award_ts = now

                            if rvol >= RVOL_HIT_THRESHOLD:
                                pts = 12.0
                            elif rvol >= RVOL_AWARD_THRESHOLD * 1.5:
                                pts = 8.0
                            else:
                                pts = 5.0

                            reason = (
                                f"intraday RVOL {rvol:.1f}x "
                                f"({wvol:,} vol last {WINDOW_MIN}m vs {int(expected_5min):,} expected)"
                            )
                            # award is synchronous (small) — run inline
                            with Session(DAEMON_NAME,
                                         note="Alpaca IEX intraday volume") as s:
                                s.award(tk, pts, reason)
                            await bus.publish("stream", {
                                "daemon": DAEMON_NAME,
                                "ticker": tk,
                                "rvol": round(rvol, 2),
                                "points": pts,
                                "reason": reason,
                                "ts": ts,
                            })

            # connection ended cleanly — loop and reconnect
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error("%s: %s", DAEMON_NAME, e)
            _write_status({"connected": False, "error": str(e)[:200]})

        if stop_event.is_set():
            break
        # backoff and retry
        log.info("%s: reconnecting in %.1fs", DAEMON_NAME, backoff)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=backoff)
            break  # stop fired during backoff
        except asyncio.TimeoutError:
            pass
        backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX)

    _write_status({"connected": False, "stopped_at":
                   datetime.now(timezone.utc).isoformat(timespec="seconds")})
    log.info("%s: stopped", DAEMON_NAME)
