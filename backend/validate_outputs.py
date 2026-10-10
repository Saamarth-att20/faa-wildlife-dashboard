"""
Check that the pipeline outputs satisfy the data contract the dashboard relies on.

Author: Saamarth Attray

Usage:
    python -m backend.validate_outputs
Exit code 0 = contract satisfied, 1 = problems found (each one is printed).
"""
import json
import sys

import pandas as pd

from backend.contract import (CLEAN_PARQUET, CLEAN_REQUIRED_COLUMNS, DAMAGE_LEVEL_ORDER, PROJECTION_PARQUET,
                              PROJECTION_REQUIRED_COLUMNS, RESULTS_JSON, RESULTS_REQUIRED_KEYS, SPECIES_GROUPS)


def validate() -> list:
    problems = []

    if not CLEAN_PARQUET.exists():
        return [f"missing {CLEAN_PARQUET}"]
    df = pd.read_parquet(CLEAN_PARQUET)
    missing = [c for c in CLEAN_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"strikes_clean.parquet is missing columns: {missing}")
    else:
        if len(df) < 50_000:
            problems.append(f"only {len(df):,} rows; the assignment needs at least 50,000")
        bad = set(df["DAMAGE_LEVEL"].dropna().unique()) - set(DAMAGE_LEVEL_ORDER)
        if bad:
            problems.append(f"DAMAGE_LEVEL has undecoded values {sorted(bad)}; expected {DAMAGE_LEVEL_ORDER}")
        bad = set(df["SPECIES_GROUP"].dropna().unique()) - set(SPECIES_GROUPS)
        if bad:
            problems.append(f"SPECIES_GROUP has values not in contract.SPECIES_GROUPS: {sorted(bad)}")
        y = df["STRIKE_CAUSED_DAMAGE"].dropna()
        if not set(y.unique()) <= {0.0, 1.0}:
            problems.append("STRIKE_CAUSED_DAMAGE must be 0.0 / 1.0 / NaN")
        unknown = df["DAMAGE_LEVEL"].isna()
        if df.loc[unknown, "STRIKE_CAUSED_DAMAGE"].notna().any():
            problems.append("STRIKE_CAUSED_DAMAGE is filled where DAMAGE_LEVEL is unknown (leave it NaN)")
        if df["INCIDENT_YEAR"].min() < 2000 or df["INCIDENT_YEAR"].max() > 2025:
            problems.append("INCIDENT_YEAR outside 2000-2025")

    if not PROJECTION_PARQUET.exists():
        problems.append(f"missing {PROJECTION_PARQUET}")
    else:
        proj = pd.read_parquet(PROJECTION_PARQUET)
        missing = [c for c in PROJECTION_REQUIRED_COLUMNS if c not in proj.columns]
        if missing:
            problems.append(f"projection_sample.parquet is missing columns: {missing}")
        if len(proj) > 25_000:
            problems.append(f"projection_sample.parquet has {len(proj):,} rows; keep it <= 20,000 so the scatter stays fast")

    if not RESULTS_JSON.exists():
        problems.append(f"missing {RESULTS_JSON}")
    else:
        try:
            res = json.loads(RESULTS_JSON.read_text())
        except json.JSONDecodeError as exc:
            return problems + [f"model_results.json is not valid JSON: {exc}"]
        missing = [k for k in RESULTS_REQUIRED_KEYS if k not in res]
        if missing:
            problems.append(f"model_results.json is missing top-level keys: {missing}")
        else:
            clf = res["classification"]
            for key in ("models", "feature_importance", "best_model", "n_test"):
                if key not in clf:
                    problems.append(f"classification.{key} missing")
            for m in clf.get("models", []):
                for key in ("name", "metrics", "confusion_matrix"):
                    if key not in m:
                        problems.append(f"classification model entry missing '{key}'")
            if len(res["normality"].get("methods", [])) < 3:
                problems.append("normality.methods must list three tests (assignment requirement)")
            if not res["randomization"].get("tests"):
                problems.append("randomization.tests is empty")
            if "profiles" not in res["clustering"]:
                problems.append("clustering.profiles missing")
    return problems


if __name__ == "__main__":
    issues = validate()
    if issues:
        print("CONTRACT PROBLEMS:")
        for i in issues:
            print("  -", i)
        sys.exit(1)
    print("OK: pipeline outputs satisfy the data contract.")
