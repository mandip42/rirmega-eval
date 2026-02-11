from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
from rirmega_eval.eval.harness import evaluate_predictions_file
from rirmega_eval.impact.ledger import ImpactLedger

DEFAULT_DATASET_DIR = Path("./dataset")  # mount or copy official artifacts here
DEFAULT_LEDGER = Path("./impact_ledger/ledger.json")


def run_eval(dataset_dir: str, task: str, split: str, pred_file) -> tuple[str, str]:
    dataset_dir_p = Path(dataset_dir)
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        evaluate_predictions_file(
            dataset_dir=dataset_dir_p,
            task=task,
            split=split,
            predictions_path=Path(pred_file),
            out_dir=out_dir,
        )

        # Optional opt-in: if ledger file exists, log
        if DEFAULT_LEDGER.exists():
            led = ImpactLedger.load(str(DEFAULT_LEDGER))
            led.log_event(event="eval_run", task=task)
            led.save(str(DEFAULT_LEDGER))

        return (str(out_dir / "metrics.json"), str(out_dir / "per_slice_metrics.csv"))


with gr.Blocks() as demo:
    gr.Markdown("# RIRMega-Eval Official Evaluator (Space)")
    dataset_dir = gr.Textbox(label="Dataset dir", value=str(DEFAULT_DATASET_DIR))
    task = gr.Dropdown(
        label="Task",
        choices=["v1_param_estimation", "v1_auralization_consistency"],
        value="v1_param_estimation",
    )
    split = gr.Dropdown(label="Split", choices=["train", "dev", "test"], value="test")
    pred = gr.File(label="Predictions file (.parquet or .jsonl)")
    btn = gr.Button("Evaluate")
    metrics_json = gr.Textbox(label="metrics.json path")
    slice_csv = gr.Textbox(label="per_slice_metrics.csv path")

    btn.click(run_eval, inputs=[dataset_dir, task, split, pred], outputs=[metrics_json, slice_csv])

demo.launch()

