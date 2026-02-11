<<<<<<< HEAD
# RIRMega-Eval (rirmega-eval)
Production-grade benchmark infrastructure for room impulse response (RIR) and acoustic ML evaluation.

This repository builds and publishes the **RIRMega-Eval** benchmark views and fixed splits from:
- RIR-Mega (HF): https://huggingface.co/datasets/mandipgoswami/rirmega :contentReference[oaicite:0]{index=0}
- RIR-Mega-Speech (HF): https://huggingface.co/datasets/mandipgoswami/rir-mega-speech :contentReference[oaicite:1]{index=1}
- Target benchmark repo (HF): https://huggingface.co/datasets/mandipgoswami/rirmega-eval (created by you)

## What you get
1) **Dataset builder** producing HF-ready artifacts:
- `data/metadata/rir_metadata.parquet`
- `data/splits/v1/{task}/{split}.txt`
- `checks/sha256sums.txt`
- `manifest.json`

2) **Canonical acoustic metrics** (documented, unit-tested):
- Schroeder EDC
- RT60 (T20/T30 selection rules)
- EDT
- DRR (standardized direct window)
- C50/C80
- Ts (center time)

3) **Official evaluation harness**:
- CLI: `scripts/evaluate.py`
- Python API: `rirmega_eval.eval.harness.evaluate_predictions(...)`

4) **Baselines**:
- Baseline 0: analytic heuristics
- Baseline 1: small ML baseline (scikit-learn)

5) **EB-1A grade adoption evidence hooks** (opt-in, privacy-safe):
- Impact Ledger (`impact_ledger/ledger.json`)
- Opt-in usage logging script
- Governance + contribution process + reproducibility checks
- Pinned deps and optional Dockerfile runtime

---

## Quickstart (Windows + PyCharm)
### 1) Create venv (Python 3.10+)
```powershell
cd rirmega-eval
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e . -c constraints.txt
=======
# rirmega-eval
>>>>>>> 81bce714f6ecad031173f91b560a05b3a69b3a3a
