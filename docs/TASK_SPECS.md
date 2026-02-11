# Task Specs (v1)

## View 1: Parameter estimation
**Task id:** `v1_param_estimation`

Input:
- RIR waveform (48 kHz assumed unless metadata specifies otherwise)

Targets (ground truth):
- `rt60_s` (seconds)
- `edt_s`
- `drr_db`
- `c50_db`
- `c80_db`
- `ts_s` (seconds)

Official scoring:
- MAE per target
- Overall score: mean of normalized MAEs (see `src/rirmega_eval/eval/scoring.py`)

## View 2: Auralization consistency
**Task id:** `v1_auralization_consistency`

Input:
- dry speech waveform
- RIR waveform

Reference:
- provided convolved speech waveform (from RIR-Mega-Speech style pairs)

Metrics:
- SI-SDR (optional)
- L2 waveform error on aligned signals (official)
- log-magnitude STFT distance (official)

Note: If RIR-Mega-Speech does not expose dry audio in the HF subset, builder switches to "reference-only" mode with convolved-only checks.

## View 3: Robustness slices
Derived evaluation slices for any view:
- `room_class` if available
- `rt60_bin`
- `distance_bin` if available
- `family` (linear vs circular arrays in RIR-Mega subset) :contentReference[oaicite:2]{index=2}
