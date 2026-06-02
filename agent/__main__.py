"""Trading agent CLI:

    python -m agent decide [--dry-run]
    python -m agent journal [--limit 50]
    python -m agent account

Defaults to dry-run for `decide` if AGENT_DRY_RUN=1 in env.
"""

from __future__ import annotations

import argparse
import json
import sys

from agent import alpaca_client, decide, journal


def cmd_decide(args):
    summary = decide.run_batch(dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, default=str))


def cmd_journal(args):
    print(json.dumps(journal.journal_payload(args.limit), indent=2, default=str))


def cmd_account(args):
    print(json.dumps({
        "is_paper":     alpaca_client.is_paper(),
        "account":      alpaca_client.get_account(),
        "positions":    alpaca_client.list_positions(),
    }, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("decide", help="run one decision pass")
    pd.add_argument("--dry-run", action="store_true")
    pd.set_defaults(func=cmd_decide)

    pj = sub.add_parser("journal", help="dump journal entries")
    pj.add_argument("--limit", type=int, default=50)
    pj.set_defaults(func=cmd_journal)

    pa = sub.add_parser("account", help="show Alpaca paper account state")
    pa.set_defaults(func=cmd_account)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
