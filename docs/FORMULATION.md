# Mathematical Formulation: Participatory Budget Allocation

## Problem
A city (Warszawa 2023, Pabulib) has a set of candidate projects. Each project has a cost and a number of
citizen votes. The city must choose which projects to fund within a fixed budget, while keeping spending
spread across thematic categories. This is a **multi-dimensional 0/1 knapsack problem (MKP)**, which is NP-hard.

## Sets and parameters
- I = {1..n}: candidate projects (n = 138 in the full instance)
- K: set of categories (primary category of each project); I_k is the set of projects in category k
- c_i (PLN): cost of project i
- v_i: votes received by project i (value / benefit proxy)
- B: total budget (30,339,246 PLN for the full instance)
- V = sum_i v_i: total votes;  V_k = sum_{i in I_k} v_i
- s = 1.25: slack factor
- C_k = min( B * (V_k / V) * s ,  sum_{i in I_k} c_i ): spending cap of category k

## Decision variables
x_i in {0,1}, i in I:  x_i = 1 if project i is funded, 0 otherwise.

## Model
maximize    sum_{i in I} v_i x_i                                   (1) total votes captured

subject to  sum_{i in I} c_i x_i <= B                              (2) overall budget
            sum_{i in I_k} c_i x_i <= C_k        for all k in K    (3) category spending caps
            x_i in {0,1}                          for all i in I   (4) binary decisions

## Notes
- With constraint (3) removed, the model is the classic 0/1 knapsack. Adding (3) gives one knapsack
  constraint per category, hence a multi-dimensional knapsack.
- Constraint (3) prevents the budget from being absorbed by one or two categories (a fairness / balance
  requirement common in participatory budgeting). Caps are proportional to each category's share of votes,
  with 25% slack.
- The solution space has 2^n points (about 2^138 for the full instance), so enumeration is impossible.
- The code also supports an optional minimum-count constraint per category, sum_{i in I_k} x_i >= m_k.
  It is not used in the reported experiments.

## Methods
1. **Exact: CP-SAT (OR-Tools).** Integer model solved to proven optimality; 8 workers, 60 s time limit.
2. **Metaheuristic: Genetic Algorithm.** Binary chromosome, tournament selection (k=3), one-point crossover
   (0.8), bit-flip mutation (0.02), elitism (2), population 100, 200 generations. Infeasible children are
   repaired by repeatedly removing the lowest votes-per-PLN project from the most violated constraint.
3. **Baseline: Greedy.** Sort by votes/cost, fund while budget and category caps allow.
