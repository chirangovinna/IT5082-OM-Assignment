from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constraints import compute_category_caps
from exact_cpsat import solve_budget_allocation
from greedy_baseline import solve_budget_allocation_greedy
from heuristic_ga import solve_budget_allocation_ga

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed"
RES = ROOT / "results"
OUT = RES / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#222222", "#777777", "#dddddd"
C_EXACT, C_GA, C_GREEDY = "#1f5fa8", "#d9822b", "#6a994e"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})


def load_or_make(name, maker):
    """Return results/<name> as a DataFrame; if absent, build it with maker(), save it, and return it."""
    path = RES / name
    if path.exists():
        return pd.read_csv(path)
    print(f"{name} not found - computing it (this can take a minute)...")
    df_new = maker()
    df_new.to_csv(path, index=False)
    return df_new


# ---- shared inputs (same pipeline as comparison.py) -------------------------
full = pd.read_csv(DATA / "full_instance.csv")
budget = float(pd.read_csv(DATA / "budgets.csv").set_index("instance").loc["full_instance", "budget"])
caps = compute_category_caps(full, budget)
exact = solve_budget_allocation(full, budget, category_caps=caps)
greedy = solve_budget_allocation_greedy(full, budget, category_caps=caps)
OPT, GREEDY = exact.total_value, greedy.total_value

INSTANCES = ["instance_small", "instance_medium", "instance_large", "full_instance"]
_budgets = pd.read_csv(DATA / "budgets.csv").set_index("instance")["budget"]


def _instance(name):
    d = pd.read_csv(DATA / f"{name}.csv")
    b = float(_budgets[name])
    return d, b, compute_category_caps(d, b)


def make_seeds_full():          # GA, 10 seeds, full instance, capped model
    rows = []
    for sd in range(10):
        g = solve_budget_allocation_ga(full, budget, category_caps=caps, seed=sd)
        rows.append(dict(seed=sd, votes=g.total_value, gap_pct=round((OPT - g.total_value) / OPT * 100, 3)))
    return pd.DataFrame(rows)


def make_convergence():         # GA convergence curve, seed 42, full instance
    g = solve_budget_allocation_ga(full, budget, category_caps=caps, seed=42)
    return pd.DataFrame({"generation": np.arange(len(g.best_fitness_per_gen)),
                         "best_votes": np.array(g.best_fitness_per_gen)})


def make_ga_per_size():         # GA, 10 seeds, every instance size
    rows = []
    for name in INSTANCES:
        d, b, c = _instance(name)
        opt = solve_budget_allocation(d, b, category_caps=c).total_value
        runs = [solve_budget_allocation_ga(d, b, category_caps=c, seed=sd) for sd in range(10)]
        vals = np.array([r.total_value for r in runs])
        gaps = (opt - vals) / opt * 100
        rows.append(dict(instance=name, n=len(d), optimum=opt, ga_mean=vals.mean(), ga_min=vals.min(),
                         ga_max=vals.max(), gap_mean=gaps.mean(), gap_min=gaps.min(), gap_max=gaps.max(),
                         n_optimal=int((vals == opt).sum()), time_mean=float(np.mean([r.runtime_sec for r in runs]))))
    return pd.DataFrame(rows)


def make_timing(repeats=15):    # CP-SAT and greedy, repeated runs, with and without caps
    rows = []
    for name in INSTANCES:
        d, b, c = _instance(name)
        for label, cc in (("no caps", None), ("with caps", c)):
            ex_t = [solve_budget_allocation(d, b, category_caps=cc).runtime_sec for _ in range(repeats)]
            gr_t = [solve_budget_allocation_greedy(d, b, category_caps=cc).runtime_sec for _ in range(repeats)]
            rows.append(dict(instance=name, n=len(d), version=label, cpsat_median=np.median(ex_t),
                             cpsat_min=min(ex_t), cpsat_max=max(ex_t), greedy_median=np.median(gr_t)))
    return pd.DataFrame(rows)



# ---- Figure 1: dataset ------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax[0].hist(np.log10(full["cost"]), bins=18, color=C_EXACT, edgecolor="white")
ax[0].set_xlabel("project cost (PLN, log10 scale)")
ax[0].set_ylabel("number of projects")
ax[0].set_title("Costs are heavily right-skewed", fontsize=9, loc="left")
ax[1].scatter(full["cost"], full["votes"], s=14, color=C_EXACT, alpha=0.75, edgecolor="white", linewidth=0.4)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("cost (PLN, log)"); ax[1].set_ylabel("votes (log)")
ax[1].set_title("Support is weakly tied to cost", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig1_data.png", dpi=200); plt.close(fig)


# ---- Figure 2: optimality gap of 10 GA seeds on the full instance -----------
seeds = load_or_make("supp_ga_seeds_full.csv", make_seeds_full).sort_values("gap_pct")
greedy_gap = (OPT - GREEDY) / OPT * 100
fig, ax = plt.subplots(figsize=(6.2, 2.6))
ax.scatter(seeds["gap_pct"], np.zeros(len(seeds)), s=55, color=C_GA, edgecolor="white", zorder=3,
           label="GA, one dot per seed (10 seeds)")
ax.axvline(0, color=C_EXACT, lw=2, label="CP-SAT (optimal, 0%)")
ax.axvline(greedy_gap, color=C_GREEDY, lw=2, ls="--", label=f"Greedy ({greedy_gap:.2f}%)")
ax.axvline(seeds["gap_pct"].mean(), color=C_GA, lw=1.2, ls=":", label=f"GA mean ({seeds['gap_pct'].mean():.2f}%)")
ax.set_yticks([]); ax.grid(axis="y", visible=False)
ax.set_xlabel("optimality gap on full instance (%)"); ax.set_xlim(-0.15, 2.1)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig2_gap_seeds.png", dpi=200); plt.close(fig)


# ---- Figure 3: GA convergence (seed 42) vs optimum and greedy ---------------
curve = load_or_make("supp_ga_convergence_seed42.csv", make_convergence)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.plot(curve["generation"], curve["best_votes"], color=C_GA, lw=2, label="GA best-so-far (seed 42)")
ax.axhline(OPT, color=C_EXACT, lw=1.6, label=f"CP-SAT optimum ({OPT:,.0f})")
ax.axhline(GREEDY, color=C_GREEDY, lw=1.6, ls="--", label=f"Greedy ({GREEDY:,.0f})")
ax.set_xlabel("generation"); ax.set_ylabel("total votes captured"); ax.set_ylim(280000, OPT * 1.009)
ax.legend(frameon=False, loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig3_ga_convergence.png", dpi=200); plt.close(fig)


# ---- Figure 4: runtime vs instance size (capped model) ----------------------
timing = load_or_make("supp_timing.csv", make_timing)
timing = timing[timing["version"] == "with caps"]
ga_time = load_or_make("supp_ga_seeds.csv", make_ga_per_size)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.plot(timing["n"], timing["cpsat_median"], "o-", color=C_EXACT, lw=2, label="CP-SAT (median of 15 runs)")
ax.plot(timing["n"], timing["greedy_median"], "s-", color=C_GREEDY, lw=2, label="Greedy (median of 15 runs)")
ax.plot(ga_time["n"], ga_time["time_mean"], "^-", color=C_GA, lw=2, label="GA (mean of 10 seeds)")
ax.set_yscale("log"); ax.set_xlabel("number of candidate projects"); ax.set_ylabel("runtime (s, log scale)")
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
fig.tight_layout(); fig.savefig(OUT / "fig4_runtime.png", dpi=200); plt.close(fig)


# ---- Figure 5: how much of each category cap the optimum uses ---------------
spend = full[full["project_id"].isin(exact.selected_ids)].groupby("primary_category")["cost"].sum()
use = pd.DataFrame({"spend": spend, "cap": pd.Series(caps)}).fillna(0)
use["pct"] = use["spend"] / use["cap"] * 100
use = use.sort_values("pct")
fig, ax = plt.subplots(figsize=(6.2, 3.1))
ax.barh(use.index, use["pct"], color=[C_EXACT if p >= 95 else "#9db7d6" for p in use["pct"]])
ax.axvline(100, color=INK, lw=1)
for i, p in enumerate(use["pct"]):
    ax.text(p + 1.5, i, f"{p:.0f}%", va="center", fontsize=8, color=INK)
ax.set_xlim(0, 115); ax.grid(axis="y", visible=False)
ax.set_xlabel("share of category cap used by the CP-SAT optimum (%)")
fig.tight_layout(); fig.savefig(OUT / "fig5_cap_usage.png", dpi=200); plt.close(fig)

