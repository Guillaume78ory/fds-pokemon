"""Builds the feature table: one row per battle, one column per feature.

Nobody needs to edit this file: Parts 2 and 3 only add functions to their own lists.
"""
import pandas as pd
from tqdm.auto import tqdm

from .config import ID_COL
from .features_static import STATIC_FEATURES
from .features_dynamic import DYNAMIC_FEATURES

ALL_FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES


def build_features(battles: list[dict], functions=None) -> pd.DataFrame:
    """Apply every feature function to every battle and return a DataFrame."""
    functions = functions or ALL_FEATURES
    rows = []
    for b in tqdm(battles, desc="features", leave=False):
        row = {ID_COL: b[ID_COL]}
        for fn in functions:
            row.update(fn(b))
        rows.append(row)
    return pd.DataFrame(rows).set_index(ID_COL)
