"""PART 2 - Team features (known before the battle starts).

Owner: Part 2. Each function takes ONE battle (dict) and returns a dict of numbers.
"""

import numpy as np
from functools import lru_cache

from .data import load_type_chart


# The six base statistics of a Pokemon
STATS = [
    "base_hp",
    "base_atk",
    "base_def",
    "base_spa",
    "base_spd",
    "base_spe"
]


# Convert a value to a valid number.
# Missing or invalid values become NaN.
def safe_float(value):
    try:
        number = float(value)

        if np.isfinite(number):
            return number

        return np.nan

    except (TypeError, ValueError):
        return np.nan


# Load the Gen 2 type chart only once
@lru_cache(maxsize=1)
def get_type_chart():
    return load_type_chart()


def calculate_type_effectiveness(
    attacking_types: list,
    defending_types: list,
    chart: dict
) -> float:
    """Calculate the best theoretical type effectiveness."""

    if not attacking_types or not defending_types:
        return 1.0

    # Ignore the placeholder type 'notype'
    attacking_types = [
        t for t in attacking_types if t != "notype"
    ]

    defending_types = [
        t for t in defending_types if t != "notype"
    ]

    if not attacking_types or not defending_types:
        return 1.0

    multipliers = []

    # Check each possible attacking type
    for attack_type in attacking_types:

        multiplier = 1.0

        # Combine effectiveness against all defending types
        for defense_type in defending_types:
            multiplier *= chart.get(
                (attack_type, defense_type), 1.0
            )

        multipliers.append(multiplier)

    # Use the best available attacking type
    return float(max(multipliers))


def baseline_team_stats(battle: dict) -> dict:
    """Calculate team statistics, lead statistics and differences."""

    feats = {}

    # Get Player 1's team
    team = battle.get("p1_team_details", []) or []

    # Calculate mean, maximum and minimum for Player 1
    for stat in STATS:

        values = [
            safe_float(p.get(stat))
            for p in team
        ]

        valid_values = [
            value for value in values
            if np.isfinite(value)
        ]

        if valid_values:
            feats[f"p1_mean_{stat}"] = float(np.mean(valid_values))
            feats[f"p1_max_{stat}"] = float(np.max(valid_values))
            feats[f"p1_min_{stat}"] = float(np.min(valid_values))

        else:
            feats[f"p1_mean_{stat}"] = np.nan
            feats[f"p1_max_{stat}"] = np.nan
            feats[f"p1_min_{stat}"] = np.nan

    # Get Player 2's starting Pokemon
    lead = battle.get("p2_lead_details", {}) or {}

    # Extract Player 2's six base statistics
    for stat in STATS:
        feats[f"p2_lead_{stat}"] = safe_float(
            lead.get(stat)
        )

    # Calculate differences between the players
    for stat in STATS:

        p1_value = feats[f"p1_mean_{stat}"]
        p2_value = feats[f"p2_lead_{stat}"]

        if np.isfinite(p1_value) and np.isfinite(p2_value):
            feats[f"diff_{stat.replace('base_', '')}"] = (
                p1_value - p2_value
            )

        else:
            feats[f"diff_{stat.replace('base_', '')}"] = np.nan

    # Calculate type advantages
    chart = get_type_chart()

    p2_types = lead.get("types", []) or []

    p1_advantages = []
    p2_advantages = []

    for pokemon in team:

        p1_types = pokemon.get("types", []) or []

        # P1 Pokemon's theoretical effectiveness against P2's lead
        p1_multiplier = calculate_type_effectiveness(
            p1_types, p2_types, chart
        )

        # P2 lead's theoretical effectiveness against this P1 Pokemon
        p2_multiplier = calculate_type_effectiveness(
            p2_types, p1_types, chart
        )

        p1_advantages.append(p1_multiplier)
        p2_advantages.append(p2_multiplier)

    # Summarise P1's type advantages
    if p1_advantages:
        feats["p1_type_mean"] = float(np.mean(p1_advantages))
        feats["p1_type_max"] = float(np.max(p1_advantages))
        feats["p1_type_min"] = float(np.min(p1_advantages))
        feats["p1_super_effective_count"] = int(
            sum(value > 1.0 for value in p1_advantages)
        )

    else:
        feats["p1_type_mean"] = np.nan
        feats["p1_type_max"] = np.nan
        feats["p1_type_min"] = np.nan
        feats["p1_super_effective_count"] = 0

    # Summarise P2's type advantages
    if p2_advantages:
        feats["p2_type_mean"] = float(np.mean(p2_advantages))
        feats["p2_type_max"] = float(np.max(p2_advantages))
        feats["p2_type_min"] = float(np.min(p2_advantages))
        feats["p2_super_effective_count"] = int(
            sum(value > 1.0 for value in p2_advantages)
        )

    else:
        feats["p2_type_mean"] = np.nan
        feats["p2_type_max"] = np.nan
        feats["p2_type_min"] = np.nan
        feats["p2_super_effective_count"] = 0

    return feats


# All static feature functions used by the project
STATIC_FEATURES = [baseline_team_stats]
