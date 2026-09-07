"""
exact_cpsat.py
Exact method: 0/1 Budget Allocation (multi-dimensional knapsack) solved
with Google OR-Tools CP-SAT.

Formulation
-----------
Decision variables:
    x_i in {0,1}   for each project i   (1 = funded)

Objective:
    maximize  sum_i  v_i * x_i          (v_i = votes, our value/benefit proxy)

Constraints:
    (1) Budget:          sum_i c_i * x_i <= B
    (2) Category cap:    sum_{i in cat k} c_i * x_i <= cat_budget_k   (optional)
    (3) Category min:    sum_{i in cat k} x_i        >= min_count_k   (optional)

CP-SAT is used instead of a pure LP-relaxation ILP solver (e.g. CBC via
PuLP) because it natively handles large numbers of binary variables and
linear constraints very efficiently, and gives us a hard time-limit with
a provable optimality gap if the limit is hit.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from ortools.sat.python import cp_model


@dataclass
class SolveResult:
    selected_ids: List[int]
    total_value: float
    total_cost: float
    runtime_sec: float
    status: str
    optimality_gap: Optional[float] = None
    n_projects: int = 0


def solve_budget_allocation(
    df: pd.DataFrame,
    budget: float,
    category_caps: Optional[Dict[str, float]] = None,
    category_min_counts: Optional[Dict[str, int]] = None,
    time_limit_sec: float = 60.0,
) -> SolveResult:
    """
    Solve the 0/1 budget allocation problem exactly with CP-SAT.

    Args:
        df: DataFrame with columns ['project_id', 'cost', 'votes', 'primary_category'].
        budget: total budget B.
        category_caps: optional {category_name: max_spend} constraints.
        category_min_counts: optional {category_name: min_num_projects} constraints.
        time_limit_sec: CP-SAT wall-clock time limit (exact result if solver
            reports OPTIMAL before this is hit; otherwise best-found + gap).
    """
    model = cp_model.CpModel()
    n = len(df)
    ids = df["project_id"].tolist()
    costs = df["cost"].astype(int).tolist()
    values = df["votes"].astype(int).tolist()
    categories = df["primary_category"].tolist()

    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

    # Constraint 1: budget
    model.Add(sum(costs[i] * x[i] for i in range(n)) <= int(budget))

    # Constraint 2 (optional): per-category spending caps
    if category_caps:
        for cat, cap in category_caps.items():
            idxs = [i for i in range(n) if categories[i] == cat]
            if idxs:
                model.Add(sum(costs[i] * x[i] for i in idxs) <= int(cap))

    # Constraint 3 (optional): per-category minimum project counts
    if category_min_counts:
        for cat, min_count in category_min_counts.items():
            idxs = [i for i in range(n) if categories[i] == cat]
            if idxs:
                model.Add(sum(x[i] for i in idxs) >= min_count)

    # Objective: maximize total value (votes) captured
    model.Maximize(sum(values[i] * x[i] for i in range(n)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 8

    start = time.perf_counter()
    status = solver.Solve(model)
    runtime = time.perf_counter() - start

    status_name = solver.StatusName(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected = [ids[i] for i in range(n) if solver.Value(x[i]) == 1]
        total_value = sum(values[i] for i in range(n) if solver.Value(x[i]) == 1)
        total_cost = sum(costs[i] for i in range(n) if solver.Value(x[i]) == 1)
        gap = None
        if status == cp_model.FEASIBLE:
            best_bound = solver.BestObjectiveBound()
            gap = (best_bound - total_value) / max(total_value, 1) * 100
    else:
        selected, total_value, total_cost, gap = [], 0, 0, None

    return SolveResult(
        selected_ids=selected,
        total_value=total_value,
        total_cost=total_cost,
        runtime_sec=runtime,
        status=status_name,
        optimality_gap=gap,
        n_projects=n,
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

    result = solve_budget_allocation(df, budget, category_caps=category_caps)
    print(f"Instance: {inst_name}  (n={result.n_projects})")
    print(f"Status: {result.status}")
    print(f"Runtime: {result.runtime_sec:.4f} s")
    print(f"Selected {len(result.selected_ids)} projects, total cost {result.total_cost:,}")
    print(f"Total value (votes) captured: {result.total_value:,}")
    if result.optimality_gap is not None:
        print(f"Optimality gap: {result.optimality_gap:.2f}%")
