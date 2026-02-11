# Contributing to RIRMega-Eval

## Ground rules
- Metrics are treated as part of the benchmark contract. Changes require review and a versioned release.
- All contributions must be reproducible and covered by tests.

## Development setup
```bash
pip install -e . -c constraints.txt
pytest -q

