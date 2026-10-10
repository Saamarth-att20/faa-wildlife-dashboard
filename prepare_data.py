"""
prepare_data.py - build the analysis CSV from the raw FAA wildlife strike export.

Author: Saamarth Attray

Keeps strikes from 2000-2025 and the 24 columns used in the CS/STAT 3654 proposal.
NUM_STRUCK is stored in the FAA export as text ranges ("1", "2-10", "11-100",
"More than 100"). It is converted to an ordinal code 1-4 so the range categories
are kept instead of being blanked out by a plain numeric conversion.

Usage:
    python prepare_data.py --raw faa_aws.csv --out data/faa_wildlife_strikes_2000_2025.csv
"""
import argparse

import pandas as pd

KEEP = [
    "INDEX_NR", "INCIDENT_DATE", "AIRPORT", "AIRPORT_ID",
    "HEIGHT", "SPEED", "DISTANCE", "NUM_ENGS", "AC_MASS",
    "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ", "NUM_STRUCK",
    "INCIDENT_MONTH", "INCIDENT_YEAR",
    "TIME_OF_DAY", "STATE", "OPERATOR", "AIRCRAFT", "TYPE_ENG",
    "PHASE_OF_FLIGHT", "SKY", "PRECIPITATION", "DAMAGE_LEVEL", "SPECIES",
]

# Ordinal code for the number of animals struck (1, 2-10, 11-100, >100)
NUM_STRUCK_CODES = {"1": 1, "2-10": 2, "11-100": 3, "More than 100": 4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="raw FAA export (faa_aws.csv)")
    parser.add_argument("--out", default="data/faa_wildlife_strikes_2000_2025.csv")
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()

    df = pd.read_csv(args.raw, usecols=KEEP, dtype={"NUM_STRUCK": "string"}, low_memory=False)
    df = df[df["INCIDENT_YEAR"].between(args.start, args.end)].copy()
    df["NUM_STRUCK"] = df["NUM_STRUCK"].str.strip().map(NUM_STRUCK_CODES).astype("float64")
    df = df[KEEP]
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows x {df.shape[1]} columns to {args.out}")
    print(df["NUM_STRUCK"].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()
