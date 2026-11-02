"""Writes a submission file in the sample_submission.csv format."""
import pandas as pd

from .config import ID_COL, TARGET


def make_submission(model, X_test: pd.DataFrame, path: str, sample: pd.DataFrame | None = None):
    pred = model.predict(X_test).astype(int)
    sub = pd.DataFrame({ID_COL: X_test.index, TARGET: pred})
    if sample is not None:
        # keep the exact order and column names of the sample file
        sub = sample[[ID_COL]].merge(sub, on=ID_COL, how="left")
        assert sub[TARGET].notna().all(), "some battle_id have no prediction"
        sub = sub[list(sample.columns)]
    sub.to_csv(path, index=False)
    print(f"[submission] {len(sub)} rows written to {path} (share of 1s: {sub[TARGET].mean():.3f})")
    return sub
