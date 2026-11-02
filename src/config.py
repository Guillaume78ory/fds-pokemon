"""Global settings shared by the whole team. Change them here only."""
from pathlib import Path

SEED = 42          # same seed everywhere -> reproducible results
N_SPLITS = 5       # number of folds for k-fold cross-validation
TARGET = "player_won"
ID_COL = "battle_id"


def find_data_dir() -> Path:
    """Return the folder that contains train.jsonl.

    - On Kaggle: searches /kaggle/input/ (the competition folder name does not matter).
    - Locally: uses ./data/ (put the downloaded files there).
    """
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hits = sorted(kaggle.rglob("train.jsonl"))
        if hits:
            return hits[0].parent
    local = Path(__file__).resolve().parent.parent / "data"
    if (local / "train.jsonl").exists():
        return local
    raise FileNotFoundError("train.jsonl not found in /kaggle/input or ./data")
