"""
constraints.py
Generates the category budget-cap constraints used by all three solvers,
turning the problem from single-constraint 0/1 knapsack into a
multi-dimensional knapsack.

"""

from typing import Dict

import pandas as pd


def compute_category_caps(df: pd.DataFrame, budget: float, slack_factor: float = 1.25) -> Dict[str, float]:

    total_votes = df["votes"].sum()
    caps = {}
    for cat, group in df.groupby("primary_category"):
        vote_share = group["votes"].sum() / total_votes
        proportional_cap = budget * vote_share * slack_factor
        # Cap can never exceed the category's own total project cost (no point
        # allocating budget a category can't physically spend).
        caps[cat] = min(proportional_cap, group["cost"].sum())
    return caps
