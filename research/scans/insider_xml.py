"""
Form 4 XML parser. Pulls the full-submission .txt at an EDGAR filing URL,
extracts the embedded <ownershipDocument> XML, and counts transactions
by type.

SEC Form 4 transaction codes (most relevant):
  P  open market / private purchase                → real bullish signal
  S  open market / private sale                    → potentially bearish (but
                                                     many are routine RSU monetisation)
  A  grant / award                                 → comp noise
  M  exercise of derivative (option exercise)      → comp noise
  F  payment of exercise price / tax withholding   → RSU vest noise
  D  sale back to issuer                           → comp noise
  G  bona-fide gift                                → noise
  C  conversion of derivative                      → comp noise
  X  exercise of in-the-money derivative           → comp noise

Acquired/Disposed flag (A/D) confirms direction.

We classify each filing by its dominant transaction:
  - "buy"   if any P code present
  - "sell"  if any S code present (and no P)
  - "noise" otherwise (comp / vest / exercise only)
"""
from __future__ import annotations
import re
from xml.etree import ElementTree as ET

# the wrapper .txt contains:   <DOCUMENT> ... <XML>  <ownershipDocument> ... </ownershipDocument>  </XML> ... </DOCUMENT>
OWNERSHIP_RE = re.compile(r"<ownershipDocument>.*?</ownershipDocument>", re.DOTALL | re.IGNORECASE)

# strip XML namespaces and stray XSL processing instructions that some
# Form 4s embed and which break ElementTree
NS_RE = re.compile(r'\sxmlns(:\w+)?="[^"]*"')


def classify_form4(submission_text: str) -> dict:
    """Return a summary of a single Form 4 filing.

    Output:
      {
        "class": "buy" | "sell" | "noise",
        "buy_shares":   int,
        "buy_dollars":  float,
        "sell_shares":  int,
        "sell_dollars": float,
        "codes":        ["P","S","A",...]   # all transaction codes seen
      }
    """
    out = {
        "class": "noise",
        "buy_shares": 0, "buy_dollars": 0.0,
        "sell_shares": 0, "sell_dollars": 0.0,
        "codes": [],
    }
    m = OWNERSHIP_RE.search(submission_text)
    if not m:
        return out
    xml_text = NS_RE.sub("", m.group(0))
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out

    # collect both non-derivative + derivative transactions
    for txn in root.iter():
        tag = txn.tag.lower()
        if tag not in ("nonderivativetransaction", "derivativetransaction"):
            continue
        code = _val(txn, "transactionCoding/transactionCode")
        if not code:
            continue
        out["codes"].append(code)
        ad = _val(txn, "transactionAmounts/transactionAcquiredDisposedCode/value")
        shares = _float(_val(txn, "transactionAmounts/transactionShares/value"))
        price  = _float(_val(txn, "transactionAmounts/transactionPricePerShare/value"))
        dollars = shares * price
        if code == "P" and ad == "A":
            out["buy_shares"]  += int(shares)
            out["buy_dollars"] += dollars
        elif code == "S" and ad == "D":
            out["sell_shares"]  += int(shares)
            out["sell_dollars"] += dollars

    if out["buy_shares"] > 0:
        out["class"] = "buy"
    elif out["sell_shares"] > 0 and not any(c == "P" for c in out["codes"]):
        out["class"] = "sell"
    return out


def _val(el, path):
    # tolerate case differences in tag names (Form 4 XML is camelCase but
    # some filers ship lowercase)
    parts = path.split("/")
    cur = el
    for p in parts:
        nxt = None
        for child in cur:
            if child.tag.lower() == p.lower():
                nxt = child
                break
        if nxt is None:
            return None
        cur = nxt
    return (cur.text or "").strip() if cur is not None else None


def _float(s):
    if s is None: return 0.0
    try: return float(s)
    except ValueError: return 0.0
