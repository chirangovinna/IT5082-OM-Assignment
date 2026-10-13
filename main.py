"""
main.py - single entry point to reproduce every result in the report.

Usage (from repo root):
    python main.py            # data prep -> comparison -> report figures
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

STEPS = [
    ("Data preparation", ["data_prep.py"]),
    ("Exact vs GA vs Greedy comparison", ["comparison.py"]),
    ("Report figures (10-seed GA study, timing)", ["report_figures.py"]),
]

for title, cmd in STEPS:
    print(f"\n=== {title} ===")
    subprocess.run([sys.executable, *cmd], cwd=SRC, check=True)
print("\nAll steps finished. See results/ for tables and figures.")
