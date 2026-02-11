# Predictions Format

Predictions are provided as:
- Parquet (`.parquet`) OR
- JSONL (`.jsonl`), one JSON object per line

Common required fields:
- `sample_id` (string)

## Task: v1_param_estimation
Required prediction columns:
- `rt60_s`, `edt_s`, `drr_db`, `c50_db`, `c80_db`, `ts_s`

## Task: v1_auralization_consistency
Allowed predictions:
- `audio_path` (path to predicted convolved waveform) OR
- `stft_mag_path` (optional advanced)

If you provide `audio_path`, evaluator loads audio and compares to reference convolved audio.

## Validation
Run:
```bash
python scripts/evaluate.py --validate_only --task v1_param_estimation --predictions preds.parquet --dataset_dir ./out

