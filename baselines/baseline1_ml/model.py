from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 300
    max_depth: int = 18
    min_samples_leaf: int = 2
    random_state: int = 1337
    n_jobs: int = -1


TARGETS: list[str] = ["rt60_s", "edt_s", "drr_db", "c50_db", "c80_db", "ts_s"]


def make_model(cfg: ModelConfig) -> MultiOutputRegressor:
    base = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    return MultiOutputRegressor(base)

