"""PART 4 - Model zoo. Classical ML only (no neural networks)."""
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED


def _logreg(C=0.1):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000, random_state=SEED))


def _xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=600, max_depth=3, learning_rate=0.02,
                         subsample=0.8, colsample_bytree=0.7,
                         min_child_weight=5, reg_lambda=5.0,
                         random_state=SEED, eval_metric="logloss")


def get_model(name: str):
    if name == "logreg":
        return _logreg(C=1.0)  # version d'origine
    if name == "logreg_c":
        return _logreg(C=0.1)  # plus de régularisation
    if name == "rf":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=500, max_depth=8,
                                                    min_samples_leaf=20,
                                                    max_features="sqrt",
                                                    random_state=SEED, n_jobs=-1))
    if name == "xgb":
        return _xgb()
    if name == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03,
                                              max_iter=400, l2_regularization=5.0,
                                              min_samples_leaf=40, random_state=SEED)
    if name == "ensemble":
        return VotingClassifier(
            estimators=[("lr", _logreg(0.1)), ("xgb", _xgb()),
                        ("hgb", get_model("hgb"))],
            voting="soft")
    raise ValueError(f"unknown model: {name}")
