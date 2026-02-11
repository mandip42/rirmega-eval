# Hugging Face Publishing

## HF token
Set token in env var and do not commit it:
- `HF_TOKEN`

## Create or update dataset repo
Use:
```bash
python scripts/publish_to_hf.py --dataset_dir ./out --hf_dataset_id mandipgoswami/rirmega-eval --hf_token_env HF_TOKEN
