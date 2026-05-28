#!/usr/bin/env python3
"""
live_daemon.py — always-on Raymond scan daemon + live dashboard server.

Replaces the GitHub Actions `live-scans.yml` cron workflows. Designed to
run as a systemd service on the Oracle alice instance and serve the
dashboard directly (no more GitHub Pages, no more commit-per-scan).

Responsibilities
----------------
1. Schedule every Phase A/B scan (the ones currently driven by
   live-scans.yml) on internal asyncio timers — no cron, no commits.
2. Run the (synchronous) scan modules in a thread pool so the event loop
   stays responsive.
3. Serve the dashboard:
   - `GET  /`                 → index.html
   - `GET  /data/live/<file>` → raw JSON files (compat with existing
                                fetch URLs in index.html)
   - `GET  /api/state`        → bundle of every live JSON in one payload
   - `GET  /api/stream`       → Server-Sent Events stream: pushes a
                                `scan` event the moment a scan finishes
                                and a `tick` keep-alive every 15s.
   - `POST /api/run/<scan>`   → manual trigger (useful while streaming)

Scans currently driven by this daemon
-------------------------------------
    macro_regime         every 15 min
    tech_slice           every 15 min, :07 offset
    reg_sho              daily 12:30 UTC
    insider_cluster      every 3h at :20
    options_unusual      every 30 min during 14:00–20:30 UTC weekdays
    material_8k          every 2h at :05
    fundamentals_snap    daily 13:15 UTC
    macro_econ           13:20, 17:20, 22:20 UTC daily

The schedule list lives in `SCHEDULES` below — adjust freely.

Phase C scans (Alpaca WS RVOL, EDGAR SC 13D/G hourly, halt tape) will be
added here once the substrate is proven.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import signal
import sys
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

try:
    from aiohttp import web
except ImportError:
    sys.stderr.write(
        "aiohttp not installed. Run: pip install aiohttp\n"
        "(also add `aiohttp>=3.9` to requirements.txt)\n"
    )
    sys.exit(2)


# ── paths ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
LIVE_DIR = REPO_ROOT / "data" / "live"
INDEX_HTML = REPO_ROOT / "index.html"

# Make the scan modules importable
sys.path.insert(0, str(REPO_ROOT))


# ── logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
log = logging.getLogger("raymond")


# ── schedule predicates ────────────────────────────────────────────────
#
# Each schedule is a callable: `next_run(after: datetime) -> datetime` that
# returns the next UTC datetime at or after `after` when the scan should
# fire. Keep them simple — these aren't full cron expressions, just the
# specific cadences we need.

UTC = timezone.utc


def _floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def every_n_minutes(n: int, offset: int = 0) -> Callable[[datetime], datetime]:
    """Every n minutes from the top of the hour, offset by `offset` min."""
    def f(after: datetime) -> datetime:
        a = _floor_minute(after)
        # find the next minute m such that (m - offset) % n == 0 and m≥a's min
        # operating on absolute epoch minutes is easiest
        epoch_min = int(a.timestamp() // 60)
        # adjust: target = ((epoch_min - offset) rounded up to multiple of n) + offset
        rel = epoch_min - offset
        k = -(-rel // n)        # ceil division
        target_min = k * n + offset
        if target_min < epoch_min or (target_min == epoch_min and after > a):
            target_min += n
        return datetime.fromtimestamp(target_min * 60, tz=UTC)
    return f


def daily_at(hour: int, minute: int) -> Callable[[datetime], datetime]:
    """Daily at HH:MM UTC."""
    def f(after: datetime) -> datetime:
        candidate = after.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= after:
            candidate += timedelta(days=1)
        return candidate
    return f


def daily_at_multi(times: list[tuple[int, int]]) -> Callable[[datetime], datetime]:
    """Daily at multiple HH:MM UTC times."""
    def f(after: datetime) -> datetime:
        best: Optional[datetime] = None
        for h, m in times:
            for day_offset in (0, 1):
                cand = (after + timedelta(days=day_offset)).replace(
                    hour=h, minute=m, second=0, microsecond=0
                )
                if cand > after and (best is None or cand < best):
                    best = cand
        assert best is not None
        return best
    return f


def every_3h_at_minute(minute: int) -> Callable[[datetime], datetime]:
    """Every 3 hours at HH:MM where HH ∈ {0,3,6,...,21}."""
    def f(after: datetime) -> datetime:
        for day_offset in (0, 1):
            base = (after + timedelta(days=day_offset)).replace(
                minute=minute, second=0, microsecond=0
            )
            for h in range(0, 24, 3):
                cand = base.replace(hour=h)
                if cand > after:
                    return cand
        raise RuntimeError("unreachable")
    return f


def every_2h_at_minute(minute: int) -> Callable[[datetime], datetime]:
    def f(after: datetime) -> datetime:
        for day_offset in (0, 1):
            base = (after + timedelta(days=day_offset)).replace(
                minute=minute, second=0, microsecond=0
            )
            for h in range(0, 24, 2):
                cand = base.replace(hour=h)
                if cand > after:
                    return cand
        raise RuntimeError("unreachable")
    return f


def market_hours_every_30min() -> Callable[[datetime], datetime]:
    """Every 30 min during 14:00–20:30 UTC, Mon–Fri (US market hours)."""
    def f(after: datetime) -> datetime:
        a = after
        for _ in range(8):  # at most a week of probing
            # candidate slots for this day
            day = a.replace(hour=0, minute=0, second=0, microsecond=0)
            if day.weekday() < 5:  # Mon–Fri
                for h in range(14, 21):
                    for m in (0, 30):
                        if h == 20 and m == 30:
                            cand = day.replace(hour=20, minute=30)
                        else:
                            cand = day.replace(hour=h, minute=m)
                        if cand > after:
                            return cand
            a = day + timedelta(days=1)
        raise RuntimeError("unreachable")
    return f


def market_hours_every_n_min(n: int) -> Callable[[datetime], datetime]:
    """Every n minutes during 14:00–20:30 UTC, Mon–Fri.

    Like market_hours_every_30min but with an arbitrary cadence. Used
    for the halt-tape scan which polls every 2 min.
    """
    def f(after: datetime) -> datetime:
        a = after
        for _ in range(8):
            day = a.replace(hour=0, minute=0, second=0, microsecond=0)
            if day.weekday() < 5:
                # generate slots at HH:00 + every n min through 20:30
                for total_min in range(14 * 60, 20 * 60 + 31, n):
                    h, m = divmod(total_min, 60)
                    cand = day.replace(hour=h, minute=m)
                    if cand > after:
                        return cand
            a = day + timedelta(days=1)
        raise RuntimeError("unreachable")
    return f


# ── scan registry ──────────────────────────────────────────────────────
@dataclass
class Scan:
    name: str
    module: str
    schedule: Callable[[datetime], datetime]
    note: str = ""
    next_run: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_run: Optional[datetime] = None
    last_result: Optional[dict] = None
    last_error: Optional[str] = None
    last_duration_sec: float = 0.0
    running: bool = False


SCHEDULES: list[Scan] = [
    Scan("macro_regime",      "research.scans.macro_regime",      every_n_minutes(15),                       "env scan"),
    Scan("tech_slice",        "research.scans.tech_slice",        every_n_minutes(15, offset=7),             "rolling 400 tickers"),
    Scan("reg_sho",           "research.scans.reg_sho",           daily_at(12, 30),                          "FINRA Reg SHO daily"),
    Scan("insider_cluster",   "research.scans.insider_cluster",   every_3h_at_minute(20),                    "EDGAR Form 4 cluster"),
    Scan("options_unusual",   "research.scans.options_unusual",   market_hours_every_30min(),                "top-watchlist options"),
    Scan("material_8k",       "research.scans.material_8k",       every_2h_at_minute(5),                     "EDGAR 8-K material events"),
    Scan("sc_13dg",           "research.scans.sc_13dg",           every_n_minutes(60, offset=35),            "EDGAR SC 13D/G poll"),
    Scan("halt_tape",         "research.scans.halt_tape",         market_hours_every_n_min(2),               "NASDAQ trading halts"),
    Scan("fundamentals_snap", "research.scans.fundamentals_snap", daily_at(13, 15),                          "Finnhub fundamentals snapshot"),
    Scan("macro_econ",        "research.scans.macro_econ",        daily_at_multi([(13, 20), (17, 20), (22, 20)]), "FRED macro series"),
]

SCANS: dict[str, Scan] = {s.name: s for s in SCHEDULES}


# ── long-running stream daemons ───────────────────────────────
# Unlike SCHEDULES (which fire on a timer), these are coroutines that
# keep a persistent connection (e.g. Alpaca websocket) open for the
# daemon's whole lifetime. Each entry is (name, module_path); the
# module must expose ``async def run(stop_event, bus, log) -> None``.
DAEMONS: list[tuple[str, str]] = [
    ("rvol_stream", "research.scans.rvol_stream"),
]


# ── SSE broadcast ──────────────────────────────────────────────────────
class Broadcaster:
    """Fan out SSE events to all connected clients."""

    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    async def publish(self, event: str, data: dict) -> None:
        payload = (event, data)
        dead: list[asyncio.Queue] = []
        for q in self._subs:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subs.discard(q)


bus = Broadcaster()

# Set when the daemon is shutting down — SSE handlers watch this so
# `runner.cleanup()` doesn't have to wait the full SIGTERM timeout for
# long-lived stream responses to close on their own.
shutdown_event: asyncio.Event | None = None


# ── scan runner ────────────────────────────────────────────────────────
async def run_scan(scan: Scan, loop: asyncio.AbstractEventLoop) -> None:
    if scan.running:
        log.warning("[%s] already running; skipping overlap", scan.name)
        return
    scan.running = True
    started = time.time()
    log.info("[%s] start", scan.name)
    try:
        mod = importlib.import_module(scan.module)
        if not hasattr(mod, "run"):
            raise RuntimeError(f"{scan.module} has no run()")
        result = await loop.run_in_executor(None, mod.run)
        scan.last_result = result if isinstance(result, dict) else {"result": result}
        scan.last_error = None
        log.info("[%s] done %s", scan.name, scan.last_result)
    except Exception as e:
        scan.last_error = f"{type(e).__name__}: {e}"
        scan.last_result = None
        log.error("[%s] FAILED: %s\n%s", scan.name, e, traceback.format_exc())
    finally:
        scan.last_duration_sec = round(time.time() - started, 2)
        scan.last_run = datetime.now(UTC)
        scan.running = False
        await bus.publish("scan", {
            "scan": scan.name,
            "started_at": (datetime.now(UTC) - timedelta(seconds=scan.last_duration_sec)).isoformat(timespec="seconds"),
            "duration_sec": scan.last_duration_sec,
            "result": scan.last_result,
            "error": scan.last_error,
        })


# ── scheduler ──────────────────────────────────────────────────────────
async def scheduler_loop() -> None:
    loop = asyncio.get_running_loop()
    # initialise next_run for every scan based on now
    now = datetime.now(UTC)
    for scan in SCHEDULES:
        scan.next_run = scan.schedule(now)
        log.info("[%s] first run at %s", scan.name, scan.next_run.isoformat())

    while True:
        now = datetime.now(UTC)
        due = [s for s in SCHEDULES if s.next_run <= now]
        for scan in due:
            asyncio.create_task(run_scan(scan, loop))
            scan.next_run = scan.schedule(now + timedelta(seconds=1))
        # sleep until the next due time (cap at 30s so manual triggers /
        # new scans show up promptly)
        upcoming = min((s.next_run for s in SCHEDULES), default=now + timedelta(seconds=30))
        delay = max(0.5, min(30.0, (upcoming - datetime.now(UTC)).total_seconds()))
        await asyncio.sleep(delay)


# ── file-watcher: emit SSE when JSON files change on disk ─────────────
async def file_watcher_loop() -> None:
    mtimes: dict[str, float] = {}
    while True:
        try:
            if LIVE_DIR.is_dir():
                for p in LIVE_DIR.glob("*.json"):
                    try:
                        mt = p.stat().st_mtime
                    except OSError:
                        continue
                    prev = mtimes.get(p.name)
                    if prev is None:
                        mtimes[p.name] = mt
                        continue
                    if mt > prev:
                        mtimes[p.name] = mt
                        await bus.publish("data", {
                            "file": p.name,
                            "mtime": mt,
                        })
        except Exception as e:
            log.warning("file_watcher error: %s", e)
        await asyncio.sleep(1.5)


# ── HTTP handlers ──────────────────────────────────────────────────────
async def handle_index(request: web.Request) -> web.Response:
    if not INDEX_HTML.exists():
        return web.Response(status=404, text="index.html missing")
    return web.FileResponse(INDEX_HTML, headers={"Cache-Control": "no-store"})


async def handle_live_file(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    if "/" in name or ".." in name:
        return web.Response(status=400, text="bad name")
    p = LIVE_DIR / name
    if not p.exists():
        return web.Response(status=404, text=f"{name} not found")
    return web.FileResponse(p, headers={
        "Cache-Control": "no-store",
        "Content-Type": "application/json",
    })


async def handle_state(request: web.Request) -> web.Response:
    bundle = {
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {},
        "scans": [
            {
                "name": s.name,
                "note": s.note,
                "next_run": s.next_run.isoformat(),
                "last_run": s.last_run.isoformat() if s.last_run else None,
                "last_duration_sec": s.last_duration_sec,
                "last_result": s.last_result,
                "last_error": s.last_error,
                "running": s.running,
            }
            for s in SCHEDULES
        ],
    }
    if LIVE_DIR.is_dir():
        for p in sorted(LIVE_DIR.glob("*.json")):
            try:
                bundle["files"][p.name] = json.loads(p.read_text())
            except Exception as e:
                bundle["files"][p.name] = {"_error": str(e)}
    return web.json_response(bundle, headers={"Cache-Control": "no-store"})


async def handle_stream(request: web.Request) -> web.StreamResponse:
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(request)
    q = bus.subscribe()
    log.info("SSE client connected (%d total)", len(bus._subs))
    try:
        # send a hello so the client sees the connection is live
        await resp.write(b": connected\n\n")
        while True:
            if shutdown_event is not None and shutdown_event.is_set():
                break
            try:
                event, data = await asyncio.wait_for(q.get(), timeout=15.0)
                payload = f"event: {event}\ndata: {json.dumps(data, separators=(',',':'))}\n\n"
                await resp.write(payload.encode())
            except asyncio.TimeoutError:
                # keep-alive ping
                await resp.write(b": ping\n\n")
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        bus.unsubscribe(q)
        log.info("SSE client disconnected (%d left)", len(bus._subs))
    return resp


async def handle_manual_run(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    scan = SCANS.get(name)
    if scan is None:
        return web.json_response({"error": f"unknown scan {name!r}"}, status=404)
    loop = asyncio.get_running_loop()
    asyncio.create_task(run_scan(scan, loop))
    return web.json_response({"ok": True, "triggered": name})


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "ok": True,
        "now": datetime.now(UTC).isoformat(timespec="seconds"),
        "sse_clients": len(bus._subs),
        "scans": [s.name for s in SCHEDULES],
    })


# ── app wiring ─────────────────────────────────────────────────────────
def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/index.html", handle_index)
    app.router.add_get("/healthz", handle_health)
    app.router.add_get("/api/state", handle_state)
    app.router.add_get("/api/stream", handle_stream)
    app.router.add_post("/api/run/{name}", handle_manual_run)
    app.router.add_get("/data/live/{name}", handle_live_file)
    # Also serve other static repo files (favicon etc.) on demand
    return app


async def main_async(host: str, port: int) -> None:
    global shutdown_event
    shutdown_event = asyncio.Event()

    app = build_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    log.info("HTTP server listening on http://%s:%d", host, port)

    sched_task = asyncio.create_task(scheduler_loop(), name="scheduler")
    watcher_task = asyncio.create_task(file_watcher_loop(), name="file_watcher")

    # ── spawn long-running stream daemons ──
    daemon_stop = asyncio.Event()
    daemon_tasks: list[asyncio.Task] = []
    for dname, mod_path in DAEMONS:
        try:
            mod = importlib.import_module(mod_path)
        except Exception as e:
            log.error("failed to import daemon %s (%s): %s", dname, mod_path, e)
            continue
        run_fn = getattr(mod, "run", None)
        if run_fn is None:
            log.error("daemon %s has no run() coroutine", dname)
            continue
        t = asyncio.create_task(run_fn(daemon_stop, bus, log), name=f"daemon:{dname}")
        daemon_tasks.append(t)
        log.info("daemon %s started", dname)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    await stop.wait()

    log.info("shutting down")
    shutdown_event.set()
    daemon_stop.set()
    # Nudge every SSE subscriber so they exit their wait_for() now rather
    # than after the 15s keep-alive timeout.
    await bus.publish("shutdown", {})
    for t in (sched_task, watcher_task, *daemon_tasks):
        t.cancel()
        with suppress(asyncio.CancelledError):
            await t
    await runner.cleanup()


def main() -> int:
    p = argparse.ArgumentParser(description="Raymond live-scan daemon")
    p.add_argument("--host", default=os.environ.get("RAYMOND_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("RAYMOND_PORT", "8420")))
    args = p.parse_args()
    try:
        asyncio.run(main_async(args.host, args.port))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
