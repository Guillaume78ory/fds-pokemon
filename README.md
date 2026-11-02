# FDS Kaggle Competition — Pokémon Battle Prediction

Predict whether Player 1 wins a Gen 2 OU battle from its first 30 turns.

## Structure
```
run.py                  end-to-end pipeline (CV + submission)
explore.py              first look at the data
kaggle_notebook.ipynb   Kaggle notebook that clones this repo and makes the 3 submissions
src/
  config.py             seed, number of folds, data path        (Part 1)
  data.py               loading the .jsonl files                 (Part 1)
  validation.py         shared stratified k-fold evaluate()      (Part 1)
  features.py           builds the feature table (do not edit)   (Part 1)
  features_static.py    team / type features                     (Part 2)
  features_dynamic.py   features from the 30 turns               (Part 3)
  models.py             models and hyperparameters               (Part 4)
  submission.py         writes the CSV in the sample format      (Part 1)
```

## Run locally
1. Put `train.jsonl`, `test.jsonl`, `sample_submission.csv` in `data/`.
2. `pip install -r requirements.txt`
3. `python explore.py`
4. `python run.py --model logreg --out submission.csv`

## Team rules
- A feature = a function `f(battle) -> dict`, added to `STATIC_FEATURES` or `DYNAMIC_FEATURES`.
- Compare everything with `src.validation.evaluate` (same folds, seed 42). Never tune on the test set.
- Work on a branch, open a pull request, merge only if `python run.py` still runs.
