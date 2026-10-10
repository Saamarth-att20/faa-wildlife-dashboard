"""
Data access for the dashboard. Reads only the files described in backend/contract.py.

Author: Saamarth Attray

Everything is loaded once per process and cached. If the back-end results are not there yet, the
accessors return None and the Modeling tab shows a "waiting for back end" message instead of crashing.
"""
import json
from functools import lru_cache

import pandas as pd

from backend.contract import (CLEAN_PARQUET, PROJECTION_PARQUET, RESULTS_JSON, SPECIES_GROUPS)

CATEGORY_COLS = ["SEASON", "STATE", "AIRPORT", "OPERATOR", "AIRCRAFT", "TYPE_ENG", "PHASE_OF_FLIGHT",
                 "TIME_OF_DAY", "SKY", "PRECIPITATION", "DAMAGE_LEVEL", "SPECIES", "SPECIES_GROUP"]

# 50 states + DC (the map cannot draw territories / non-US codes that appear in the FAA data)
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY",
    "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

# Columns the "feature explorer" can describe: label -> (column, unit)
NUMERIC_EXPLORER = {
    "Height above ground (ft)": ("HEIGHT", "ft"),
    "Speed (knots)": ("SPEED", "kt"),
    "Distance from airport (nm)": ("DISTANCE", "nm"),
    "Aircraft mass class (1-5)": ("AC_MASS", ""),
    "Number of engines": ("NUM_ENGS", ""),
    "Animals struck (estimated)": ("NUM_STRUCK", ""),
    "Repair cost, inflation-adj. ($)": ("COST_REPAIRS_INFL_ADJ", "$"),
    "Other cost, inflation-adj. ($)": ("COST_OTHER_INFL_ADJ", "$"),
}
CATEGORICAL_EXPLORER = {
    "Time of day": "TIME_OF_DAY", "Phase of flight": "PHASE_OF_FLIGHT", "Sky condition": "SKY",
    "Engine type": "TYPE_ENG", "Damage level": "DAMAGE_LEVEL", "Species group": "SPECIES_GROUP",
    "Season": "SEASON", "State": "STATE", "Operator": "OPERATOR", "Aircraft": "AIRCRAFT", "Species": "SPECIES",
}


@lru_cache(maxsize=1)
def get_df() -> pd.DataFrame:
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(
            f"{CLEAN_PARQUET} not found. Run `python -m backend.pipeline` first (see README.md).")
    df = pd.read_parquet(CLEAN_PARQUET)
    for col in CATEGORY_COLS:
        if col in df:
            df[col] = df[col].astype("category")
    df["INCIDENT_YEAR"] = df["INCIDENT_YEAR"].astype("int16")
    df["INCIDENT_MONTH"] = df["INCIDENT_MONTH"].astype("int8")
    return df


@lru_cache(maxsize=1)
def get_results():
    """model_results.json as a dict, or None if the back end has not produced it yet."""
    if not RESULTS_JSON.exists():
        return None
    return json.loads(RESULTS_JSON.read_text())


@lru_cache(maxsize=1)
def get_projection():
    if not PROJECTION_PARQUET.exists():
        return None
    return pd.read_parquet(PROJECTION_PARQUET)


def filter_df(states=None, groups=None, years=None) -> pd.DataFrame:
    """Subset used by the Trends tab. Empty / None selections mean 'no filter'."""
    df = get_df()
    mask = pd.Series(True, index=df.index)
    if years:
        mask &= df["INCIDENT_YEAR"].between(years[0], years[1])
    if states:
        mask &= df["STATE"].isin(states)
    if groups:
        mask &= df["SPECIES_GROUP"].isin(groups)
    return df[mask]


def state_options():
    states = get_df()["STATE"].dropna().astype(str).value_counts()
    return [{"label": f"{s}  ({n:,})", "value": s} for s, n in states.items()]


def species_group_options():
    present = set(get_df()["SPECIES_GROUP"].dropna().astype(str))
    return [{"label": g, "value": g} for g in SPECIES_GROUPS if g in present]


def damage_rate(frame: pd.DataFrame) -> float:
    """Share of strikes with a KNOWN outcome that caused damage."""
    known = frame["STRIKE_CAUSED_DAMAGE"].dropna()
    return float(known.mean()) if len(known) else float("nan")
