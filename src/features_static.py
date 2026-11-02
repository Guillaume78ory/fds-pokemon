"""PART 2 - Team features (known before the battle starts).

Owner: Part 2. Each function takes ONE battle (dict) and returns a dict of numbers.
Add your functions to STATIC_FEATURES at the bottom.
"""
import numpy as np

STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]


def baseline_team_stats(battle: dict) -> dict:
    """Mean of each base stat over Player 1's 6 Pokemon + Player 2 lead's stats."""
    feats = {}
    team = battle.get("p1_team_details", []) or []
    for s in STATS:
        vals = [p.get(s, np.nan) for p in team]
        feats[f"p1_mean_{s}"] = float(np.nanmean(vals)) if vals else np.nan
    lead = battle.get("p2_lead_details", {}) or {}
    for s in STATS:
        feats[f"p2_lead_{s}"] = lead.get(s, np.nan)
    feats["diff_spe"] = feats["p1_mean_base_spe"] - feats["p2_lead_base_spe"]
    return feats


STATIC_FEATURES = [baseline_team_stats]
