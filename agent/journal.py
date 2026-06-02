"""
Trading-agent journal — sqlite-backed log of decisions, outcomes, reflections.

All entries are scoped to one db file (default: data/agent.db). The
/api/agent-journal endpoint in live_daemon.py reads from here.

Schema is created on first connection and is idempotent.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "agent.db"

# Allowed action labels — matches what pitches.html's loadAgentJournal expects
ACTIONS = {"SKIP", "SMALL", "NORMAL", "CONVICTION", "EXIT"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            INTEGER NOT NULL,
    ticker        TEXT    NOT NULL,
    action        TEXT    NOT NULL,
    size_usd      REAL    NOT NULL DEFAULT 0,
    rationale     TEXT,
    bup_score     REAL,
    pct_30d       REAL,
    context_json  TEXT,
    order_id      TEXT,
    dry_run       INTEGER NOT NULL DEFAULT 0,
    status        TEXT    NOT NULL DEFAULT 'open'    -- open | filled | rejected | exited
);

CREATE TABLE IF NOT EXISTS outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id     INTEGER NOT NULL REFERENCES decisions(id),
    ts              INTEGER NOT NULL,
    realized_pnl    REAL,
    pnl_pct         REAL,
    exit_reason     TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS reflections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         INTEGER NOT NULL,
    summary    TEXT,
    playbook   TEXT
);

CREATE TABLE IF NOT EXISTS shadow_trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_entry      INTEGER NOT NULL,
    ticker        TEXT    NOT NULL,
    entry_price   REAL    NOT NULL,
    notional_usd  REAL    NOT NULL,
    ts_exit       INTEGER,
    exit_price    REAL,
    pnl_pct       REAL,
    status        TEXT    NOT NULL DEFAULT 'open',   -- open | closed | failed
    note          TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_ts          ON decisions(ts DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_ticker_ts   ON decisions(ticker, ts DESC);
CREATE INDEX IF NOT EXISTS idx_outcomes_decision     ON outcomes(decision_id);
CREATE INDEX IF NOT EXISTS idx_shadow_status         ON shadow_trades(status, ts_entry);
CREATE INDEX IF NOT EXISTS idx_shadow_ticker         ON shadow_trades(ticker, ts_entry DESC);

-- FTS5 index over decision rationale + ticker for episodic memory.
-- Kept in sync via triggers so episodes.find_similar() can MATCH against
-- past decisions.
CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
    ticker, rationale, action UNINDEXED,
    content='decisions', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS decisions_ai AFTER INSERT ON decisions BEGIN
    INSERT INTO decisions_fts(rowid, ticker, rationale, action)
    VALUES (new.id, new.ticker, COALESCE(new.rationale,''), new.action);
END;
CREATE TRIGGER IF NOT EXISTS decisions_ad AFTER DELETE ON decisions BEGIN
    INSERT INTO decisions_fts(decisions_fts, rowid, ticker, rationale, action)
    VALUES('delete', old.id, old.ticker, COALESCE(old.rationale,''), old.action);
END;
CREATE TRIGGER IF NOT EXISTS decisions_au AFTER UPDATE ON decisions BEGIN
    INSERT INTO decisions_fts(decisions_fts, rowid, ticker, rationale, action)
    VALUES('delete', old.id, old.ticker, COALESCE(old.rationale,''), old.action);
    INSERT INTO decisions_fts(rowid, ticker, rationale, action)
    VALUES (new.id, new.ticker, COALESCE(new.rationale,''), new.action);
END;
"""


def _db_path() -> Path:
    p = Path(os.environ.get("AGENT_DB", str(DEFAULT_DB)))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or _db_path()
    conn = sqlite3.connect(path, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
    finally:
        conn.close()


# ── writes ────────────────────────────────────────────────────────────

def record_decision(*, ticker: str, action: str, size_usd: float,
                    rationale: str, bup_score: float | None = None,
                    pct_30d: float | None = None,
                    context: dict | None = None,
                    order_id: str | None = None,
                    dry_run: bool = False,
                    status: str = "open") -> int:
    """Insert a decision row and return its id."""
    action = action.upper()
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")
    with connect() as c:
        cur = c.execute(
            """INSERT INTO decisions
               (ts, ticker, action, size_usd, rationale, bup_score,
                pct_30d, context_json, order_id, dry_run, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (int(time.time()), ticker.upper(), action, float(size_usd or 0),
             rationale, bup_score, pct_30d,
             json.dumps(context, default=str) if context else None,
             order_id, 1 if dry_run else 0, status),
        )
        return int(cur.lastrowid)


def record_outcome(*, decision_id: int, realized_pnl: float | None = None,
                   pnl_pct: float | None = None, exit_reason: str = "",
                   notes: str = "") -> int:
    with connect() as c:
        cur = c.execute(
            """INSERT INTO outcomes
               (decision_id, ts, realized_pnl, pnl_pct, exit_reason, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (decision_id, int(time.time()), realized_pnl, pnl_pct,
             exit_reason, notes),
        )
        c.execute("UPDATE decisions SET status = 'exited' WHERE id = ?",
                  (decision_id,))
        return int(cur.lastrowid)


def update_decision_status(decision_id: int, status: str,
                           order_id: str | None = None) -> None:
    with connect() as c:
        if order_id is not None:
            c.execute("UPDATE decisions SET status=?, order_id=? WHERE id=?",
                      (status, order_id, decision_id))
        else:
            c.execute("UPDATE decisions SET status=? WHERE id=?",
                      (status, decision_id))


def record_reflection(summary: str, playbook: str) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT INTO reflections (ts, summary, playbook) VALUES (?, ?, ?)",
            (int(time.time()), summary, playbook),
        )
        return int(cur.lastrowid)


# ── reads ─────────────────────────────────────────────────────────────

def recent_decisions(limit: int = 200) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM decisions ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def recent_outcomes(limit: int = 200) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            """SELECT o.*, d.ticker, d.action
                 FROM outcomes o
                 JOIN decisions d ON d.id = o.decision_id
                ORDER BY o.ts DESC LIMIT ?""", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def latest_reflection() -> dict | None:
    with connect() as c:
        row = c.execute(
            "SELECT * FROM reflections ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def all_open_decisions() -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM decisions WHERE status IN ('open','filled') ORDER BY ts DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def journal_payload(limit: int = 400) -> dict:
    """Single payload the UI consumes."""
    decisions = recent_decisions(limit)
    outcomes  = recent_outcomes(limit)
    refl      = latest_reflection()
    entries: list[dict] = []
    for d in decisions:
        entries.append({
            "kind":      "decision",
            "id":        d["id"],
            "ts":        d["ts"],
            "ticker":    d["ticker"],
            "action":    d["action"],
            "size_usd":  d["size_usd"],
            "rationale": d["rationale"],
            "bup_score": d["bup_score"],
            "pct_30d":   d["pct_30d"],
            "status":    d["status"],
            "order_id":  d["order_id"],
            "dry_run":   bool(d["dry_run"]),
        })
    for o in outcomes:
        entries.append({
            "kind":         "outcome",
            "id":           o["id"],
            "decision_id":  o["decision_id"],
            "ts":           o["ts"],
            "ticker":       o["ticker"],
            "pnl_pct":      o["pnl_pct"],
            "realized_pnl": o["realized_pnl"],
            "exit_reason":  o["exit_reason"],
            "notes":        o["notes"],
        })
    if refl:
        entries.append({
            "kind":     "reflection",
            "id":       refl["id"],
            "ts":       refl["ts"],
            "summary":  refl["summary"],
        })
    entries.sort(key=lambda e: e["ts"], reverse=True)
    return {
        "entries":  entries[:limit],
        "playbook": refl["playbook"] if refl else "",
        "count":    len(entries),
    }


def already_decided_recently(ticker: str, window_sec: int = 60 * 60 * 6) -> bool:
    """True if we already wrote a decision for this ticker in the last window."""
    cutoff = int(time.time()) - window_sec
    with connect() as c:
        row = c.execute(
            "SELECT 1 FROM decisions WHERE ticker=? AND ts>=? LIMIT 1",
            (ticker.upper(), cutoff),
        ).fetchone()
        return row is not None


# ── decisions awaiting outcomes (used by outcomes_tracker) ────────────

def open_buy_decisions() -> list[dict]:
    """Filled BUY decisions that don't have an outcome row yet."""
    with connect() as c:
        rows = c.execute(
            """SELECT d.* FROM decisions d
                LEFT JOIN outcomes o ON o.decision_id = d.id
               WHERE d.status = 'filled'
                 AND d.dry_run = 0
                 AND d.action IN ('SMALL','NORMAL','CONVICTION')
                 AND o.id IS NULL
               ORDER BY d.ts ASC"""
        ).fetchall()
        return [dict(r) for r in rows]


def decisions_outcomes_since(since_ts: int) -> tuple[list[dict], list[dict]]:
    """Returns (decisions, outcomes) joined for reflection."""
    with connect() as c:
        dec = c.execute(
            "SELECT * FROM decisions WHERE ts >= ? ORDER BY ts ASC", (since_ts,)
        ).fetchall()
        out = c.execute(
            """SELECT o.*, d.ticker, d.action, d.size_usd, d.rationale
                 FROM outcomes o JOIN decisions d ON d.id = o.decision_id
                WHERE o.ts >= ? ORDER BY o.ts ASC""", (since_ts,)
        ).fetchall()
        return [dict(r) for r in dec], [dict(r) for r in out]


# ── shadow ledger ─────────────────────────────────────────────────────

def shadow_open(ticker: str, entry_price: float, notional_usd: float,
                note: str = "") -> int:
    with connect() as c:
        cur = c.execute(
            """INSERT INTO shadow_trades
               (ts_entry, ticker, entry_price, notional_usd, status, note)
               VALUES (?, ?, ?, ?, 'open', ?)""",
            (int(time.time()), ticker.upper(),
             float(entry_price), float(notional_usd), note),
        )
        return int(cur.lastrowid)


def shadow_close(shadow_id: int, exit_price: float, note: str = "") -> None:
    with connect() as c:
        row = c.execute(
            "SELECT entry_price FROM shadow_trades WHERE id=?", (shadow_id,)
        ).fetchone()
        if not row:
            return
        ep = float(row["entry_price"])
        pnl_pct = ((exit_price - ep) / ep) * 100 if ep else 0.0
        c.execute(
            """UPDATE shadow_trades
                  SET ts_exit=?, exit_price=?, pnl_pct=?, status='closed',
                      note = COALESCE(note,'') || CASE WHEN ?='' THEN '' ELSE ' | '||? END
                WHERE id=?""",
            (int(time.time()), float(exit_price), pnl_pct, note, note, shadow_id),
        )


def shadow_open_trades(older_than_sec: int = 0) -> list[dict]:
    cutoff = int(time.time()) - older_than_sec
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM shadow_trades WHERE status='open' AND ts_entry<=? ORDER BY ts_entry ASC",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]


def shadow_recent_tickers(window_sec: int) -> set[str]:
    cutoff = int(time.time()) - window_sec
    with connect() as c:
        rows = c.execute(
            "SELECT DISTINCT ticker FROM shadow_trades WHERE ts_entry>=?",
            (cutoff,),
        ).fetchall()
        return {r["ticker"] for r in rows}


def shadow_stats() -> dict:
    with connect() as c:
        closed = c.execute(
            "SELECT pnl_pct, notional_usd FROM shadow_trades WHERE status='closed'"
        ).fetchall()
        open_n = c.execute(
            "SELECT COUNT(*) AS n FROM shadow_trades WHERE status='open'"
        ).fetchone()["n"]
    if not closed:
        return {"closed": 0, "open": open_n, "avg_pnl_pct": None,
                "win_rate": None, "total_pnl_usd": 0.0}
    pnls = [float(r["pnl_pct"] or 0) for r in closed]
    realized = sum((float(r["pnl_pct"] or 0) / 100.0) * float(r["notional_usd"] or 0)
                   for r in closed)
    wins = sum(1 for p in pnls if p > 0)
    return {
        "closed":         len(closed),
        "open":           open_n,
        "avg_pnl_pct":    sum(pnls) / len(pnls),
        "win_rate":       wins / len(pnls),
        "total_pnl_usd":  realized,
    }


# ── episodic memory (FTS5) ────────────────────────────────────────────

def find_similar_decisions(query: str, limit: int = 5,
                           exclude_ticker: str | None = None) -> list[dict]:
    """FTS5 lookup over past decisions. Returns rows with score (lower=better)."""
    if not query or not query.strip():
        return []
    # Sanitize for FTS5 — strip non-alphanumerics, keep words ≥3 chars
    import re as _re
    tokens = [t for t in _re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", query)]
    if not tokens:
        return []
    fts_q = " OR ".join(tokens[:12])
    sql = """SELECT d.*, fts.rank AS score
               FROM decisions_fts fts
               JOIN decisions d ON d.id = fts.rowid
               LEFT JOIN outcomes o ON o.decision_id = d.id
              WHERE decisions_fts MATCH ?
                {extra}
              ORDER BY fts.rank
              LIMIT ?"""
    extra = "AND d.ticker != ?" if exclude_ticker else ""
    params: tuple
    if exclude_ticker:
        params = (fts_q, exclude_ticker.upper(), limit)
    else:
        params = (fts_q, limit)
    with connect() as c:
        try:
            rows = c.execute(sql.format(extra=extra), params).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]


def get_outcome_for(decision_id: int) -> dict | None:
    with connect() as c:
        r = c.execute(
            "SELECT * FROM outcomes WHERE decision_id=? ORDER BY id DESC LIMIT 1",
            (decision_id,),
        ).fetchone()
        return dict(r) if r else None
