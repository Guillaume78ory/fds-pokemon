"""PART 4 - Model zoo. Classical ML only (no neural networks).

Add or tune models here. Each entry is a scikit-learn compatible estimator.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED


def get_model(name: str):
    if name == "logreg":
        # Imputer: fills missing values; Scaler: puts all features on the same scale.
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=SEED))
    if name == "rf":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                             eval_metric="logloss")
    raise ValueError(f"unknown model: {name}")
