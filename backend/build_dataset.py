"""
Rebuild the 24-column project CSV (2000-2025) from the full FAA Wildlife Strike export.

Author: Saamarth Attray

Why this exists: the first CSV we made converted NUM_STRUCK to a number with errors="coerce", which
blanked every range label ("2-10", "11-100", "More than 100") and left a column that was just 1 or NaN.
This script keeps those rows by converting each range to an estimated count (its midpoint; open-ended
"More than 100" uses its lower bound + 1).

Usage:
    python -m backend.build_dataset path/to/faa_aws.csv
Output:
    data/raw/faa_wildlife_strikes_2000_2025.csv   (285,875 rows, same 24 columns as before)
"""
import sys

import pandas as pd

from backend.contract import RAW_CSV

KEEP_COLUMNS = [
    "INDEX_NR", "INCIDENT_DATE", "AIRPORT", "AIRPORT_ID",
    "HEIGHT", "SPEED", "DISTANCE", "NUM_ENGS", "AC_MASS",
    "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ", "NUM_STRUCK",
    "INCIDENT_MONTH", "INCIDENT_YEAR",
    "TIME_OF_DAY", "STATE", "OPERATOR", "AIRCRAFT", "TYPE_ENG",
    "PHASE_OF_FLIGHT", "SKY", "PRECIPITATION", "DAMAGE_LEVEL", "SPECIES",
]

# Estimated number of animals for each range label in the FAA export
NUM_STRUCK_MIDPOINTS = {"1": 1.0, "2-10": 6.0, "11-100": 55.5, "More than 100": 101.0}


def build(source_csv: str, out_csv=RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(source_csv, low_memory=False)
    df = df[(df["INCIDENT_YEAR"] >= 2000) & (df["INCIDENT_YEAR"] <= 2025)].copy()
    df["NUM_STRUCK"] = df["NUM_STRUCK"].astype("string").str.strip().map(NUM_STRUCK_MIDPOINTS).astype(float)
    df = df[KEEP_COLUMNS]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {len(df):,} rows x {len(df.columns)} cols -> {out_csv}")
    print(df["NUM_STRUCK"].value_counts(dropna=False).sort_index())
    return df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m backend.build_dataset path/to/faa_aws.csv")
    build(sys.argv[1])
