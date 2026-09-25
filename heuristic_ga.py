"""
heuristic_ga.py
Heuristic method: Genetic Algorithm for the 0/1 Budget Allocation problem.

Chromosome:  binary vector x of length n (1 = project funded)
Fitness:     total value (votes) captured, subject to budget feasibility
Repair:      infeasible individuals (over budget) are repaired by greedily
             dropping the worst value/cost-ratio projects until feasible,
             rather than just penalizing fitness -- this keeps the search
             inside the feasible region, which converges much faster for
             knapsack-style problems than penalty-based fitness.

This is a self-contained implementation (no DEAP/PyGAD dependency) so the
selection/crossover/mutation/repair logic is fully transparent for the
report and viva.
"""

import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class GAResult:
    selected_ids: List[int]
    total_value: float
    total_cost: float
    runtime_sec: float
    best_fitness_per_gen: List[float]
    n_projects: int = 0


def _repair(
    chromosome: np.ndarray,
    costs: np.ndarray,
    values: np.ndarray,
    budget: float,
    categories: np.ndarray = None,
    category_caps: dict = None,
) -> np.ndarray:
    """
    Repair an infeasible individual by dropping projects until all constraints
    are satisfied: the overall budget AND, if provided, every category cap.

    Strategy: repeatedly find the *worst offending* violated constraint
    (budget or a specific category), and within it drop the project with the
    lowest value/cost ratio, until every constraint holds. This is a greedy
    repair -- it doesn't guarantee the "best" feasible neighbor, but it's fast
    and keeps the GA firmly inside the feasible region each generation.
    """
    chromosome = chromosome.copy()
    category_caps = category_caps or {}

    def violated_constraints():
        selected = chromosome == 1
        violations = []
        total_cost = costs[selected].sum()
        if total_cost > budget:
            violations.append(("budget", None, total_cost - budget))
        if categories is not None:
            for cat, cap in category_caps.items():
                cat_mask = selected & (categories == cat)
                cat_cost = costs[cat_mask].sum()
                if cat_cost > cap:
                    violations.append(("category", cat, cat_cost - cap))
        return violations

    violations = violated_constraints()
    max_iters = int(chromosome.sum()) + 1  # safety bound
    it = 0
    while violations and it < max_iters:
        it += 1
        # Address the largest violation first
        violations.sort(key=lambda v: v[2], reverse=True)
        kind, cat, _ = violations[0]

        selected = chromosome == 1
        if kind == "budget":
            candidate_mask = selected
        else:
            candidate_mask = selected & (categories == cat)

        candidate_idx = np.where(candidate_mask)[0]
        if len(candidate_idx) == 0:
            break  # nothing left to drop for this constraint; give up gracefully

        ratio = values[candidate_idx] / np.maximum(costs[candidate_idx], 1)
        worst = candidate_idx[np.argmin(ratio)]
        chromosome[worst] = 0

        violations = violated_constraints()

    return chromosome


def _fitness(chromosome: np.ndarray, values: np.ndarray) -> float:
    return float(np.sum(values[chromosome == 1]))


def _tournament_select(pop: np.ndarray, fitnesses: np.ndarray, k: int = 3) -> np.ndarray:
    idxs = np.random.choice(len(pop), size=k, replace=False)
    best = idxs[np.argmax(fitnesses[idxs])]
    return pop[best].copy()


def _crossover(parent1: np.ndarray, parent2: np.ndarray, rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() > rate:
        return parent1.copy(), parent2.copy()
    point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def _mutate(chromosome: np.ndarray, rate: float = 0.02) -> np.ndarray:
    mask = np.random.random(len(chromosome)) < rate
    chromosome = chromosome.copy()
    chromosome[mask] = 1 - chromosome[mask]
    return chromosome


def solve_budget_allocation_ga(
    df: pd.DataFrame,
    budget: float,
    category_caps: dict = None,
    population_size: int = 100,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.02,
    elitism: int = 2,
    seed: int = 42,
) -> GAResult:
    """
    Solve the 0/1 budget allocation problem with a Genetic Algorithm.

    Args:
        df: DataFrame with columns ['project_id', 'cost', 'votes', 'primary_category'].
        budget: total budget B.
        category_caps: optional {category_name: max_spend} constraints (multi-
            dimensional knapsack). If None, only the overall budget applies.
        population_size, generations, crossover_rate, mutation_rate, elitism: GA hyperparameters.
        seed: RNG seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    ids = df["project_id"].to_numpy()
    costs = df["cost"].to_numpy(dtype=float)
    values = df["votes"].to_numpy(dtype=float)
    categories = df["primary_category"].to_numpy() if category_caps else None
    n = len(df)

    start = time.perf_counter()

    # Initialize population: random binary vectors, then repaired to be feasible
    population = np.random.randint(0, 2, size=(population_size, n))
    population = np.array(
        [_repair(ind, costs, values, budget, categories, category_caps) for ind in population]
    )

    best_fitness_per_gen = []
    best_overall = population[0].copy()
    best_overall_fitness = _fitness(best_overall, values)

    for gen in range(generations):
        fitnesses = np.array([_fitness(ind, values) for ind in population])

        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_overall_fitness:
            best_overall_fitness = fitnesses[gen_best_idx]
            best_overall = population[gen_best_idx].copy()
        best_fitness_per_gen.append(best_overall_fitness)

        # Elitism: carry the top individuals forward unchanged
        elite_idx = np.argsort(fitnesses)[-elitism:]
        new_population = [population[i].copy() for i in elite_idx]

        while len(new_population) < population_size:
            parent1 = _tournament_select(population, fitnesses)
            parent2 = _tournament_select(population, fitnesses)
            child1, child2 = _crossover(parent1, parent2, crossover_rate)
            child1 = _mutate(child1, mutation_rate)
            child2 = _mutate(child2, mutation_rate)
            child1 = _repair(child1, costs, values, budget, categories, category_caps)
            child2 = _repair(child2, costs, values, budget, categories, category_caps)
            new_population.extend([child1, child2])

        population = np.array(new_population[:population_size])

    runtime = time.perf_counter() - start

    selected_ids = [int(ids[i]) for i in range(n) if best_overall[i] == 1]
    total_cost = float(np.sum(costs[best_overall == 1]))

    return GAResult(
        selected_ids=selected_ids,
        total_value=best_overall_fitness,
        total_cost=total_cost,
        runtime_sec=runtime,
        best_fitness_per_gen=best_fitness_per_gen,
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

    result = solve_budget_allocation_ga(df, budget, category_caps=category_caps)
    print(f"Instance: {inst_name}  (n={result.n_projects})")
    print(f"Runtime: {result.runtime_sec:.4f} s")
    print(f"Selected {len(result.selected_ids)} projects, total cost {result.total_cost:,.0f}")
    print(f"Total value (votes) captured: {result.total_value:,.0f}")
