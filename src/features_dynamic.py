"""PART 3 - Battle features (what happens in the first 30 turns).

Owner: Part 3. Each function takes ONE battle (dict) and returns a dict of numbers.
Add your functions to DYNAMIC_FEATURES at the bottom.
"""
import numpy as np


def baseline_last_turn(battle: dict) -> dict:
    """HP of the active Pokemon at the last visible turn + number of fainted Pokemon."""
    timeline = battle.get("battle_timeline", []) or []
    if not timeline:
        return {"p1_last_hp": np.nan, "p2_last_hp": np.nan, "p1_fainted": 0, "p2_fainted": 0}
    last = timeline[-1]
    p1 = last.get("p1_pokemon_state", {}) or {}
    p2 = last.get("p2_pokemon_state", {}) or {}
    p1_fnt = {t.get("p1_pokemon_state", {}).get("name") for t in timeline
              if (t.get("p1_pokemon_state") or {}).get("status") == "fnt"}
    p2_fnt = {t.get("p2_pokemon_state", {}).get("name") for t in timeline
              if (t.get("p2_pokemon_state") or {}).get("status") == "fnt"}
    return {
        "p1_last_hp": p1.get("hp_pct", np.nan),
        "p2_last_hp": p2.get("hp_pct", np.nan),
        "p1_fainted": len(p1_fnt),
        "p2_fainted": len(p2_fnt),
    }


DYNAMIC_FEATURES = [baseline_last_turn]
