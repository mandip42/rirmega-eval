# Baselines

## Baseline 0: heuristics
File: `baselines/baseline0_heuristics.py`

Produces:
- parameter estimation predictions using the canonical metric implementations directly on the RIR waveform.
This is a sanity-check baseline. It is not “learning”, but it validates the evaluation pipeline.

## Baseline 1: small ML baseline (scikit-learn)
Folder: `baselines/baseline1_ml/`

- Features: low-cost time and spectral features from RIR
- Model: RandomForestRegressor (multi-output)
- Fixed seeds, fixed train/dev splits

Run:
```bash
python baselines/baseline1_ml/train.py --dataset_dir ./out_rirmega_eval --task v1_param_estimation

