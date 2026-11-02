"""PART 1 - Shared validation. Everyone uses evaluate() so scores are comparable.

Never look at the test set to make decisions: only these cross-validation scores.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from .config import SEED, N_SPLITS


def get_folds(y, n_splits: int = N_SPLITS, seed: int = SEED):
    """Same folds for everybody (stratified = same win/loss ratio in each fold)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def evaluate(model, X: pd.DataFrame, y, n_splits: int = N_SPLITS, seed: int = SEED,
             verbose: bool = True) -> dict:
    """Train/score the model on each fold. Returns mean and std accuracy.

    The model is cloned for each fold, so the object you pass is never modified.
    """
    y = np.asarray(y)
    scores, oof = [], np.zeros(len(y))
    for k, (tr, va) in enumerate(get_folds(y, n_splits, seed)):
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        pred = m.predict(X.iloc[va])
        oof[va] = pred
        scores.append(accuracy_score(y[va], pred))
        if verbose:
            print(f"  fold {k + 1}/{n_splits}: accuracy = {scores[-1]:.4f}")
    res = {"mean": float(np.mean(scores)), "std": float(np.std(scores)),
           "folds": scores, "oof": oof}
    if verbose:
        print(f"  => CV accuracy = {res['mean']:.4f} +/- {res['std']:.4f}")
    return res
