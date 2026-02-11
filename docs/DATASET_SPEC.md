# RIRMega-Eval Dataset Spec

## Output folder layout
Builder writes to:
- `data/metadata/rir_metadata.parquet`
- `data/splits/v1/{task}/{split}.txt`
- `checks/sha256sums.txt`
- `manifest.json`

## Metadata parquet schema (minimum contract)
Required columns:
- `sample_id` (string): unique per benchmark example
- `view` (string): `v1_param_estimation` or `v1_auralization_consistency`
- `source_dataset` (string): `rirmega` or `rir-mega-speech`
- `rir_id` (string)
- `rir_ref` (string): HF path/id or local path for core export
- `fs` (int32)
- `room_id` (string, optional but recommended)
- `family` (string, optional): linear/circular :contentReference[oaicite:3]{index=3}

Targets for view 1 (float32, may contain NaN):
- `rt60_s`, `edt_s`, `drr_db`, `c50_db`, `c80_db`, `ts_s`

For view 2:
- `dry_ref` (string, optional)
- `convolved_ref` (string, optional)
- `pair_id` (string, optional)

## Split files
Each split file is a newline-delimited list of `sample_id`.

## Manifest
`manifest.json` includes:
- version
- build mode (core/full)
- source dataset refs
- counts
- schema version
- metric config defaults
- checksums file path

