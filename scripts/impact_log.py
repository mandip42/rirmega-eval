from __future__ import annotations

import argparse

from rirmega_eval.impact.ledger import ImpactLedger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", type=str, required=True, choices=["eval_run", "build_run", "publish_run"])
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--ledger_path", type=str, default="impact_ledger/ledger.json")
    args = ap.parse_args()

    ledger = ImpactLedger.load(args.ledger_path)
    ledger.log_event(event=args.event, task=args.task)
    ledger.save(args.ledger_path)
    print("Logged.")


if __name__ == "__main__":
    main()

