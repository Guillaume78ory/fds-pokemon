"""End-to-end pipeline: load -> features -> cross-validation -> fit on all train -> submission.

Usage:  python run.py --model logreg --out submission_logreg.csv
"""
import argparse

from src.data import load_data, get_target
from src.features import build_features
from src.models import get_model
from src.submission import make_submission
from src.validation import evaluate


def main(model_name: str, out: str):
    train, test, sample = load_data()
    X_train, y = build_features(train), get_target(train)
    X_test = build_features(test)[X_train.columns]
    print(f"[features] {X_train.shape[1]} features")

    model = get_model(model_name)
    print(f"[cv] {model_name}")
    evaluate(model, X_train, y)

    model.fit(X_train, y)           # final model trained on ALL training data
    make_submission(model, X_test, out, sample)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="logreg")
    p.add_argument("--out", default="submission.csv")
    a = p.parse_args()
    main(a.model, a.out)
