"""
data_prep.py
Loads the parsed Pabulib dataset and prepares instances of varying size
for the comparative exact-vs-heuristic evaluation.

Produces:
    data/processed/full_instance.csv       - all 138 projects (heuristic scalability test)
    data/processed/instance_small.csv      - 15 projects  (exact ILP - fast, exhaustive-ish)
    data/processed/instance_medium.csv     - 30 projects  (exact ILP - still tractable)
    data/processed/instance_large.csv      - 60 projects  (exact ILP starts to slow down)
    data/processed/summary_stats.txt       - quick text summary for the report

Each instance CSV also carries its own scaled-down `budget` (proportional
to the subset's total cost), so smaller instances remain a meaningful
selection problem rather than "just fund everything".
"""

from pathlib import Path

import pandas as pd

from pb_parser import parse_pb

RANDOM_SEED = 42
INSTANCE_SIZES = {"instance_small": 15, "instance_medium": 30, "instance_large": 60}
# Fraction of a subset's total project cost to use as that subset's budget cap.
# Kept the same fraction as the full instance (budget / total_cost) so difficulty
# (selectivity) is comparable across instance sizes.
BUDGET_FRACTION = None  # computed from the full instance at runtime


def build_instances(raw_pb_path: str, out_dir: str = "data/processed") -> dict:
    meta, projects, _ = parse_pb(raw_pb_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_budget = meta["budget"]
    full_cost_total = projects["cost"].sum()
    budget_fraction = full_budget / full_cost_total

    # Full instance (for heuristic scalability testing)
    full_df = projects.copy()
    full_df.attrs["budget"] = full_budget
    full_df.to_csv(out_dir / "full_instance.csv", index=False)

    written = {"full_instance": (full_df, full_budget)}

    # Smaller random subsets (for exact-method runtime comparison)
    rng_state = RANDOM_SEED
    for name, size in INSTANCE_SIZES.items():
        subset = projects.sample(n=size, random_state=rng_state).reset_index(drop=True)
        subset_budget = round(subset["cost"].sum() * budget_fraction)
        subset.to_csv(out_dir / f"{name}.csv", index=False)
        written[name] = (subset, subset_budget)
        rng_state += 1  # vary seed slightly per size so subsets differ

    # Write a companion budgets file (small csv doesn't carry .attrs on reload)
    budgets = pd.DataFrame(
        [{"instance": k, "n_projects": len(v[0]), "budget": v[1]} for k, v in written.items()]
    )
    budgets.to_csv(out_dir / "budgets.csv", index=False)

    _write_summary(meta, projects, written, out_dir)
    return written


def _write_summary(meta, projects, written, out_dir):
    lines = []
    lines.append(f"Source instance: {meta.get('unit')} {meta.get('instance')}")
    lines.append(f"Full budget: {meta.get('budget'):,} {meta.get('currency')}")
    lines.append(f"Full project count: {len(projects)}")
    lines.append(f"Total cost of all candidate projects: {projects['cost'].sum():,}")
    lines.append(
        f"Actual (greedy-rule) selection: {int(projects['selected'].sum())} projects, "
        f"cost {projects.loc[projects['selected'] == 1, 'cost'].sum():,}, "
        f"total votes captured {projects.loc[projects['selected'] == 1, 'votes'].sum():,}"
    )
    lines.append("")
    lines.append("Prepared instances:")
    for name, (df, budget) in written.items():
        lines.append(
            f"  {name:16s} n={len(df):4d}  budget={budget:>12,.0f}  "
            f"total_cost={df['cost'].sum():>14,.0f}  total_votes={df['votes'].sum():>10,.0f}"
        )
    text = "\n".join(lines)
    (out_dir / "summary_stats.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    path_arg = sys.argv[1] if len(sys.argv) > 1 else "data/Poland_Warszawa_2023.pb"
    path = Path(path_arg)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path_arg

    build_instances(path, out_dir=PROJECT_ROOT / "data/processed")
