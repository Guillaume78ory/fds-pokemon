"""Loading the competition data."""
import json
from pathlib import Path

import pandas as pd

from .config import find_data_dir, TARGET, ID_COL


def read_jsonl(path: Path) -> list[dict]:
    """Read a .jsonl file: one JSON object (= one battle) per line."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_data(data_dir: Path | None = None):
    """Load train battles, test battles and the sample submission.

    Returns
    -------
    train : list[dict]   battles WITH the target 'player_won'
    test  : list[dict]   battles WITHOUT the target
    sample: pd.DataFrame sample_submission.csv (format to respect)
    """
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    train = read_jsonl(data_dir / "train.jsonl")
    test = read_jsonl(data_dir / "test.jsonl")
    sample_path = data_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path) if sample_path.exists() else None
    print(f"[data] {len(train)} train battles, {len(test)} test battles from {data_dir}")
    return train, test, sample


def get_target(battles: list[dict]) -> pd.Series:
    """Extract the target (1 = player 1 wins) as integers."""
    return pd.Series([int(b[TARGET]) for b in battles], name=TARGET)


def get_ids(battles: list[dict]) -> pd.Series:
    return pd.Series([b[ID_COL] for b in battles], name=ID_COL)


def load_type_chart(data_dir: Path | None = None) -> dict:
    """Gen 2 type chart -> {(attacking_type, defending_type): multiplier}.

    Pairs not listed in the file are neutral (1.0): use chart.get((a, d), 1.0).
    The type 'notype' (second slot of single-type Pokemon) is always neutral.
    """
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    df = pd.read_csv(data_dir / "type_chart_gen2.csv")
    return {(r.attacking_type, r.defending_type): float(r.multiplier) for r in df.itertuples()}
