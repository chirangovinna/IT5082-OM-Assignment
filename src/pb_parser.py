"""
pb_parser.py
Parses Pabulib (.pb) participatory budgeting files into clean pandas
DataFrames ready for optimization modeling (ILP / CP-SAT / GA / SA).

Pabulib format reference: http://pabulib.org
A .pb file has three sections, each starting with a bare header line:
    META
    key;value
    ...
    PROJECTS
    project_id;cost;votes;...
    ...
    VOTES
    voter_id;...
    ...

Usage:
    from pb_parser import parse_pb

    meta, projects = parse_pb("data/Poland_Warszawa_2023.pb")
    print(meta["budget"], len(projects))
"""

import csv
import io
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def parse_pb(filepath: str, load_votes: bool = False) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Parse a Pabulib .pb file.

    Args:
        filepath: path to the .pb file.
        load_votes: if True, also parse and return the VOTES section
                    (can be large — skip if you only need META + PROJECTS).

    Returns:
        meta: dict of metadata key -> value (numeric fields auto-cast to int/float)
        projects: DataFrame of all projects (one row per project)
        votes: DataFrame of votes (empty DataFrame if load_votes=False)
    """
    filepath = Path(filepath)
    raw = filepath.read_text(encoding="utf-8")

    # Split into sections by the bare section-header lines
    sections = {"META": [], "PROJECTS": [], "VOTES": []}
    current = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped in sections:
            current = stripped
            continue
        if current is not None and stripped != "":
            sections[current].append(line)

    # --- META ---
    meta_rows = list(csv.reader(sections["META"], delimiter=";"))
    header = meta_rows[0]  # ["key", "value"]
    meta = {}
    for row in meta_rows[1:]:
        if len(row) < 2:
            continue
        key, value = row[0], row[1]
        meta[key] = _coerce_numeric(value)

    # --- PROJECTS ---
    projects_text = "\n".join(sections["PROJECTS"])
    projects = pd.read_csv(io.StringIO(projects_text), sep=";", dtype=str)
    projects = _clean_projects(projects)

    # --- VOTES (optional, can be large) ---
    votes = pd.DataFrame()
    if load_votes and sections["VOTES"]:
        votes_text = "\n".join(sections["VOTES"])
        votes = pd.read_csv(io.StringIO(votes_text), sep=";", dtype=str)

    return meta, projects, votes


def _coerce_numeric(value: str):
    """Cast META values to int/float where possible, else leave as string."""
    try:
        if "." in value:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        return value


def _clean_projects(df: pd.DataFrame) -> pd.DataFrame:
    """Cast known numeric columns and tidy up the projects table."""
    numeric_cols = ["project_id", "cost", "votes", "selected", "latitude", "longitude"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # project_id and selected are integer-like
    if "project_id" in df.columns:
        df["project_id"] = df["project_id"].astype("Int64")
    if "selected" in df.columns:
        df["selected"] = df["selected"].astype("Int64")

    # category is a comma-separated multi-label field — keep as string,
    # but also provide a `primary_category` convenience column
    if "category" in df.columns:
        df["primary_category"] = (
            df["category"].fillna("uncategorized").str.split(",").str[0].str.strip()
        )

    return df.reset_index(drop=True)


def summarize(meta: Dict, projects: pd.DataFrame) -> None:
    """Print a quick sanity-check summary — useful as a first data-prep step."""
    print(f"Instance:        {meta.get('unit')} {meta.get('instance')}")
    print(f"Total budget:    {meta.get('budget'):,} {meta.get('currency', '')}")
    print(f"Num projects:    {len(projects)} (meta says {meta.get('num_projects')})")
    print(f"Total cost of ALL projects: {projects['cost'].sum():,}")
    print(f"Num selected (actual, greedy rule): {int(projects['selected'].sum())}")
    print(f"Cost of actually-selected projects: {projects.loc[projects['selected'] == 1, 'cost'].sum():,}")
    print(f"Categories: {sorted(projects['primary_category'].unique())}")
    print()
    print(projects[["project_id", "cost", "votes", "primary_category", "selected"]].head())


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_warszawa2023.pb"
    meta, projects, _ = parse_pb(path)
    summarize(meta, projects)
