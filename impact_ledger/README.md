# Impact Ledger (Opt-in)

Goal: provide privacy-safe adoption signals suitable for public transparency and “major significance” evidence.

## What it stores
`impact_ledger/ledger.json` stores:
- total event counts
- unique user approximations (salted hash)
- unique run ids (salted hash)
- last seen timestamps
- per-task usage counts

## Privacy model
- No raw identifiers, no IP, no HF username, no email.
- We hash a locally generated install id with a local salt.
- Logging is opt-in only.

## How to log
```bash
python scripts/impact_log.py --event eval_run --task v1_param_estimation

