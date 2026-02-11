# RIRMega-Eval Space (Optional)

This folder contains a minimal Gradio app that:
- lets users pick task + split
- upload predictions parquet
- runs official evaluation
- shows results and a submission bundle hash
- optionally logs opt-in Impact Ledger events

Deploy:
1) Create a Hugging Face Space (Gradio).
2) Push `space/` contents as the Space repo root.
3) Ensure Space has persistent storage enabled if you want `impact_ledger/ledger.json` to persist.

The Space uses only local files and does not require HF tokens for evaluation.
