"""
constraints.py
Generates the category budget-cap constraints used by all three solvers,
turning the problem from single-constraint 0/1 knapsack into a
multi-dimensional knapsack.

Policy rule: each category's budget cap is proportional to that category's
share of total community votes within the instance, scaled by a slack
factor so the problem stays feasible but still meaningfully constrained.
This models a realistic municipal policy: "spend roughly in proportion to
demonstrated citizen demand across categories," rather than letting the
optimizer dump the entire budget into whichever 1-2 categories have the
best value/cost ratio.
"""

from typing import Dict

import pandas as pd


def compute_category_caps(df: pd.DataFrame, budget: float, slack_factor: float = 1.25) -> Dict[str, float]:
    """
    Compute a per-category budget cap proportional to the category's vote share.

    Args:
        df: DataFrame with columns ['votes', 'primary_category', 'cost'].
        budget: total budget B.
        slack_factor: multiplier > 1 applied to the strict proportional share,
            so caps aren't so tight they force infeasibility or degenerate
            solutions. 1.25 means each category can spend up to 125% of its
            "fair share" based on vote demand.

    Returns:
        {category_name: cap_amount}
    """
    total_votes = df["votes"].sum()
    caps = {}
    for cat, group in df.groupby("primary_category"):
        vote_share = group["votes"].sum() / total_votes
        proportional_cap = budget * vote_share * slack_factor
        # Cap can never exceed the category's own total project cost (no point
        # allocating budget a category can't physically spend).
        caps[cat] = min(proportional_cap, group["cost"].sum())
    return caps
