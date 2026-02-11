from __future__ import annotations

import pandas as pd
import pytest
from rirmega_eval.eval.harness import validate_predictions_file


def test_predictions_validation(tmp_path):
    p = tmp_path / "preds.parquet"
    df = pd.DataFrame({"sample_id": ["a", "b"], "rt60_s": [0.3, 0.4], "edt_s": [0.2, 0.3], "drr_db": [5.0, 4.0], "c50_db": [2.0, 1.0], "c80_db": [3.0, 2.0], "ts_s": [0.01, 0.02]})
    df.to_parquet(p, index=False)
    validate_predictions_file(p, task="v1_param_estimation")

    df2 = pd.DataFrame({"sample_id": ["a"]})
    p2 = tmp_path / "bad.parquet"
    df2.to_parquet(p2, index=False)
    with pytest.raises(ValueError):
        validate_predictions_file(p2, task="v1_param_estimation")

