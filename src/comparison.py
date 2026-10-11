"""
comparison.py
Runs the exact CP-SAT method, the GA heuristic, and the greedy baseline
across all prepared instances.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from exact_cpsat import solve_budget_allocation
from greedy_baseline import solve_budget_allocation_greedy
from heuristic_ga import solve_budget_allocation_ga
from constraints import compute_category_caps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTANCES = ["instance_small", "instance_medium", "instance_large", "full_instance"]


def run_all():
    data_dir = PROJECT_ROOT / "data/processed"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    budgets = pd.read_csv(data_dir / "budgets.csv")

    rows = []
    ga_convergence_curves = {}

    for inst in INSTANCES:
        df = pd.read_csv(data_dir / f"{inst}.csv")
        budget = float(budgets.loc[budgets["instance"] == inst, "budget"].iloc[0])
        category_caps = compute_category_caps(df, budget)

        exact = solve_budget_allocation(df, budget, category_caps=category_caps)
        ga = solve_budget_allocation_ga(df, budget, category_caps=category_caps)
        greedy = solve_budget_allocation_greedy(df, budget, category_caps=category_caps)

        ga_convergence_curves[inst] = ga.best_fitness_per_gen

        for method, res in [("CP-SAT (exact)", exact), ("Genetic Algorithm", ga), ("Greedy", greedy)]:
            total_value = res.total_value
            gap_vs_optimal = (
                (exact.total_value - total_value) / exact.total_value * 100
                if exact.total_value > 0
                else None
            )
            rows.append(
                {
                    "instance": inst,
                    "n_projects": len(df),
                    "method": method,
                    "total_value": total_value,
                    "total_cost": res.total_cost,
                    "n_selected": len(res.selected_ids),
                    "runtime_sec": res.runtime_sec,
                    "gap_vs_optimal_pct": round(gap_vs_optimal, 3) if gap_vs_optimal is not None else 0.0,
                }
            )

        print(f"Done: {inst}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_dir / "comparison_table.csv", index=False)
    print("\n" + results_df.to_string(index=False))

    _plot_quality_vs_size(results_df, results_dir)
    _plot_runtime_vs_size(results_df, results_dir)
    _plot_ga_convergence(ga_convergence_curves, results_dir)
    _compare_vs_actual(data_dir, results_dir, budgets)

    return results_df


def _plot_quality_vs_size(results_df, results_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in results_df["method"].unique():
        sub = results_df[results_df["method"] == method].sort_values("n_projects")
        ax.plot(sub["n_projects"], sub["total_value"], marker="o", label=method)
    ax.set_xlabel("Number of candidate projects (instance size)")
    ax.set_ylabel("Total value captured (votes)")
    ax.set_title("Solution Quality vs. Instance Size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "quality_vs_size.png", dpi=150)
    plt.close(fig)


def _plot_runtime_vs_size(results_df, results_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in results_df["method"].unique():
        sub = results_df[results_df["method"] == method].sort_values("n_projects")
        ax.plot(sub["n_projects"], sub["runtime_sec"], marker="o", label=method)
    ax.set_xlabel("Number of candidate projects (instance size)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime vs. Instance Size")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(results_dir / "runtime_vs_size.png", dpi=150)
    plt.close(fig)


def _plot_ga_convergence(curves, results_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for inst, curve in curves.items():
        ax.plot(curve, label=inst)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (total value / votes)")
    ax.set_title("GA Convergence Across Instance Sizes")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "ga_convergence.png", dpi=150)
    plt.close(fig)


def _compare_vs_actual(data_dir, results_dir, budgets):
    """Compare our methods' picks on the FULL instance against Warszawa's real (greedy-rule) selection."""
    df = pd.read_csv(data_dir / "full_instance.csv")
    budget = float(budgets.loc[budgets["instance"] == "full_instance", "budget"].iloc[0])
    category_caps = compute_category_caps(df, budget)

    exact = solve_budget_allocation(df, budget, category_caps=category_caps)
    ga = solve_budget_allocation_ga(df, budget, category_caps=category_caps)

    actual_selected = set(df.loc[df["selected"] == 1, "project_id"].tolist())
    actual_value = df.loc[df["selected"] == 1, "votes"].sum()
    actual_cost = df.loc[df["selected"] == 1, "cost"].sum()

    rows = [
        {
            "method": "Actual (Warszawa city rule)",
            "total_value": actual_value,
            "total_cost": actual_cost,
            "n_selected": len(actual_selected),
            "overlap_with_actual": len(actual_selected),
        },
        {
            "method": "CP-SAT (exact)",
            "total_value": exact.total_value,
            "total_cost": exact.total_cost,
            "n_selected": len(exact.selected_ids),
            "overlap_with_actual": len(set(exact.selected_ids) & actual_selected),
        },
        {
            "method": "Genetic Algorithm",
            "total_value": ga.total_value,
            "total_cost": ga.total_cost,
            "n_selected": len(ga.selected_ids),
            "overlap_with_actual": len(set(ga.selected_ids) & actual_selected),
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "vs_actual_selection.csv", index=False)
    print("\nComparison vs. actual city selection (full instance):")
    print(out.to_string(index=False))


if __name__ == "__main__":
    run_all()
