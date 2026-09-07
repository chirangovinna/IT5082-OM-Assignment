"""
greedy_baseline.py
Classic greedy heuristic baseline for 0/1 knapsack-style budget allocation:
sort projects by value/cost ratio (votes per PLN) descending, and fund
projects in that order until the budget is exhausted.

Included as a third comparison point alongside the exact CP-SAT method and
the Genetic Algorithm -- it's fast and simple, but doesn't backtrack, so it
generally leaves a quality gap relative to both.
"""

import time
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class GreedyResult:
    selected_ids: List[int]
    total_value: float
    total_cost: float
    runtime_sec: float
    n_projects: int = 0


def solve_budget_allocation_greedy(df: pd.DataFrame, budget: float, category_caps: dict = None) -> GreedyResult:
    start = time.perf_counter()

    work = df.copy()
    work["ratio"] = work["votes"] / work["cost"].replace(0, 1)
    work = work.sort_values("ratio", ascending=False)

    category_caps = category_caps or {}
    category_spend = {cat: 0.0 for cat in category_caps}

    selected_ids, total_cost, total_value = [], 0.0, 0.0
    for _, row in work.iterrows():
        cat = row.get("primary_category")
        cat_cap = category_caps.get(cat)
        cat_ok = cat_cap is None or (category_spend.get(cat, 0.0) + row["cost"] <= cat_cap)
        if cat_ok and total_cost + row["cost"] <= budget:
            selected_ids.append(int(row["project_id"]))
            total_cost += row["cost"]
            total_value += row["votes"]
            if cat is not None and cat in category_spend:
                category_spend[cat] += row["cost"]

    runtime = time.perf_counter() - start

    return GreedyResult(
        selected_ids=selected_ids,
        total_value=total_value,
        total_cost=total_cost,
        runtime_sec=runtime,
        n_projects=len(df),
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "data/processed/instance_small.csv"
    csv_path = Path(csv_arg)
    if not csv_path.is_absolute() and not csv_path.exists():
        csv_path = PROJECT_ROOT / csv_arg

    df = pd.read_csv(csv_path)
    budgets = pd.read_csv(PROJECT_ROOT / "data/processed/budgets.csv")
    inst_name = csv_path.stem
    budget = float(budgets.loc[budgets["instance"] == inst_name, "budget"].iloc[0])

    from constraints import compute_category_caps
    category_caps = compute_category_caps(df, budget)

    result = solve_budget_allocation_greedy(df, budget, category_caps=category_caps)
    print(f"Instance: {inst_name}  (n={result.n_projects})")
    print(f"Runtime: {result.runtime_sec:.4f} s")
    print(f"Selected {len(result.selected_ids)} projects, total cost {result.total_cost:,.0f}")
    print(f"Total value (votes) captured: {result.total_value:,.0f}")
