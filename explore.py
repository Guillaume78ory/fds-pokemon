"""PART 1 - Step 3: first look at the data. Run: python explore.py"""
import json
from collections import Counter

from src.data import load_data, get_target

train, test, sample = load_data()
b = train[0]

print("\n=== Keys of one battle ===")
for k, v in b.items():
    print(f"  {k}: {type(v).__name__}" + (f" (len {len(v)})" if isinstance(v, (list, dict)) else f" = {v}"))

print("\n=== One Pokemon of Player 1 ===")
print(json.dumps(b["p1_team_details"][0], indent=2))
print("\n=== Player 2 lead ===")
print(json.dumps(b["p2_lead_details"], indent=2))
print("\n=== First turn of the timeline ===")
print(json.dumps(b["battle_timeline"][0], indent=2))

y = get_target(train)
print(f"\n=== Target === share of wins for P1: {y.mean():.3f}  (0.5 = balanced)")
lens = Counter(len(x["battle_timeline"]) for x in train)
print("Timeline lengths (turns: count):", dict(sorted(lens.items())))
if sample is not None:
    print("\nsample_submission.csv:\n", sample.head())
