"""
EDGAR SC 13D / SC 13G poll — institutional 5%+ ownership disclosures.

Anyone (institution, fund, activist) who crosses 5% beneficial ownership
of a US equity must file:

  SC 13D    "active" intent — activist / control-seeking. 10-day deadline
            from crossing 5%. Strong signal: an entity now intends to
            influence management.
  SC 13D/A  amendment to an existing SC 13D (position change, new intent
            statement, group composition change).
  SC 13G    "passive" 5%+ holder — qualified institutions (mutual funds,
            insurers) certifying they will NOT seek control. Quarterly
            cadence on top of crossing-5% trigger.
  SC 13G/A  amendment (annual update or material change).

Why this matters for raymond:
  - SC 13D filings frequently precede activist campaigns and rerating.
  - Clusters of SC 13G filings around earnings indicate institutional
    accumulation. Either way, the disclosure itself is the catalyst.

Scoring tiers reflect the gap in informational content:
  - SC 13D is the strongest single-filing signal we get from EDGAR.
  - SC 13G is informational but not directional — score it lower.

Implementation note: like material_8k, this only reads the EDGAR daily
form-index. We do NOT fetch the body of each filing here — that would
let us extract the filer name and stake size, but it 30× the request
volume against SEC. v1 just tags the issuer; v2 could fetch the bodies
for top-watchlist tickers only.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from research.scans.insider_cluster import (
    _load_cik_to_ticker, _business_days, _fetch_form_index,
)
from research.live_score_engine import Session

SCAN_NAME = "sc_13dg"
LOOKBACK_DAYS = 2   # tight window — SC 13D/G are uncommon

# award thresholds — SC 13D is much stronger than SC 13G
PTS_13D = 12.0            # activist / 5%+ active stake — banner-eligible
PTS_13D_AMEND = 7.0       # amendment: stake size change or new intent
PTS_13G = 5.0             # passive 5%+ holder disclosure
PTS_13G_AMEND = 3.0       # routine amendment
CLUSTER_BONUS = 4.0       # ≥3 13D/G filings at one issuer in the window


# Map of EDGAR form types we care about to (points, label) tuples.
# The daily index reports forms verbatim, e.g. "SC 13D", "SC 13D/A".
FORM_AWARDS: dict[str, tuple[float, str]] = {
    "SC 13D":   (PTS_13D,       "SC 13D filed — 5%+ active stake"),
    "SC 13D/A": (PTS_13D_AMEND, "SC 13D/A amendment"),
    "SC 13G":   (PTS_13G,       "SC 13G filed — 5%+ passive holder"),
    "SC 13G/A": (PTS_13G_AMEND, "SC 13G/A amendment"),
}


def _parse_form_index_for_13dg(text: str) -> list[tuple[str, int]]:
    """Return (form_type, cik) tuples for SC 13D/G filings in this index."""
    out: list[tuple[str, int]] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("-----"):
            in_table = True
            continue
        if not in_table:
            continue
        # "SC 13D" / "SC 13D/A" / "SC 13G" / "SC 13G/A" — the form occupies
        # the first 12 chars of the row (fixed-width). The daily index has
        # space-separated form tokens, so "SC 13D" is two tokens.
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[0] != "SC":
            continue
        form = f"{parts[0]} {parts[1]}"   # e.g. "SC 13D" or "SC 13D/A"
        if form not in FORM_AWARDS:
            continue
        cik = None
        for tok in parts[2:]:
            if tok.isdigit():
                cik = int(tok)
                break
        if cik:
            out.append((form, cik))
    return out


def run() -> dict:
    cik_map = _load_cik_to_ticker()
    if not cik_map:
        with Session(SCAN_NAME, note="EDGAR SC 13D/G poll") as s:
            s.log("failed to load SEC ticker→CIK map", level="info")
            return {"ok": False, "reason": "no_ticker_map"}

    with Session(SCAN_NAME, note="EDGAR SC 13D/G poll") as s:
        today = datetime.now(timezone.utc).date()
        days = _business_days(today - timedelta(days=1), LOOKBACK_DAYS)
        # (cik, form) -> count of that form for that issuer in the window
        per_cik_form: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        days_with_data = 0
        for d in days:
            txt = _fetch_form_index(d)
            if not txt:
                continue
            days_with_data += 1
            for form, cik in _parse_form_index_for_13dg(txt):
                per_cik_form[cik][form] += 1

        total_filings = sum(sum(v.values()) for v in per_cik_form.values())
        s.log(f"loaded {days_with_data}/{LOOKBACK_DAYS} daily form indexes", level="info")
        s.log(f"saw {total_filings} SC 13D/G filings across {len(per_cik_form)} issuers", level="info")

        awarded = 0
        for cik, forms in per_cik_form.items():
            ticker = cik_map.get(cik)
            if not ticker:
                continue

            # pick the strongest form filed for this issuer — that's the
            # headline. Counts of weaker forms still feed cluster_bonus.
            best_form = max(forms.keys(), key=lambda f: FORM_AWARDS[f][0])
            pts, label = FORM_AWARDS[best_form]
            extras = [f"{n}× {f}" for f, n in forms.items() if f != best_form or n > 1]
            reason = label
            if extras:
                reason += f" (plus {', '.join(extras)})"

            total = sum(forms.values())
            if total >= 3:
                pts += CLUSTER_BONUS
                reason += f" — {total} filings clustered"

            s.award(ticker, pts, reason)
            awarded += 1

        return {
            "ok": True,
            "days_with_data": days_with_data,
            "issuers_seen": len(per_cik_form),
            "filings_seen": total_filings,
            "awarded": awarded,
        }


if __name__ == "__main__":
    print(run())
