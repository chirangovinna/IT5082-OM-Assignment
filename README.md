# IT5082 Optimization Methods Assignment: Participatory Budget Allocation

Selecting which citizen projects to fund under a fixed budget and category spending caps
(a multi-dimensional 0/1 knapsack problem), using real data from the **Warszawa 2023** participatory
budget (Pabulib, `data/Poland_Warszawa_2023.pb`).

## Team
| Member | Student ID | Email | Contribution |
|---|---|---|---|
| S. Thanushan | MS26912684 | ms26912684@my.sliit.lk | Data pipeline, constraint generation, greedy baseline, mathematical formulation, report and video assembly |
| G.C. Devinda | MS26902616 | ms26902616@my.sliit.lk | Genetic Algorithm, exact CP-SAT model, comparative evaluation, report and video assembly |

## Methods
| Method | File | Type |
|---|---|---|
| CP-SAT (OR-Tools) | `src/exact_cpsat.py` | Exact |
| Genetic Algorithm with repair | `src/heuristic_ga.py` | Metaheuristic |
| Value/cost greedy | `src/greedy_baseline.py` | Baseline heuristic |

Formulation: see `docs/FORMULATION.md`.

## Repository layout
```
data/                Raw .pb file and processed instances (small 15, medium 30, large 60, full 138)
src/pb_parser.py     Pabulib parser
src/data_prep.py     Builds instances and budgets
src/constraints.py   Category cap generation
src/exact_cpsat.py   Exact solver
src/heuristic_ga.py  Genetic algorithm
src/greedy_baseline.py
src/comparison.py    Runs all methods on all instances
src/report_figures.py  Multi-seed study, timing, report figures
results/             CSV tables and figures
main.py              Runs the whole pipeline
```

## Setup and run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                                       # full pipeline
# or individually (run from src/):
cd src
python exact_cpsat.py ../data/processed/instance_small.csv
python heuristic_ga.py
python greedy_baseline.py
```

## Key results (full instance, 138 projects)
| Method | Votes | Gap to optimum | Runtime |
|---|---|---|---|
| CP-SAT (exact) | 371,661 | 0% | 0.12 s |
| Genetic Algorithm (seed 42) | 368,787 | 0.77% | 13.9 s |
| Greedy | 368,634 | 0.81% | 0.02 s |
| Actual city rule | 290,845 | 21.7% | n/a |

Over 10 GA seeds the mean gap is 1.01% (range 0.07% to 1.82%). On the 15-, 30- and 60-project instances all
three methods reach the optimum.
