from __future__ import annotations

import pandas as pd
from rirmega_eval.io.splits import make_splits_grouped


def test_grouped_split_unseen_group():
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(30)],
            "room_id": ["r0"] * 10 + ["r1"] * 10 + ["r2"] * 10,
        }
    )
    splits = make_splits_grouped(df, group_col="room_id", seed=1337, ratios=(0.7, 0.1, 0.2), require_unseen_group_test=True)
    tr = set(splits["train"])
    dv = set(splits["dev"])
    te = set(splits["test"])
    assert tr.isdisjoint(te)
    assert dv.isdisjoint(te)

