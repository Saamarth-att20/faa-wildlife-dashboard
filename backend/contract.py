"""
Shared data contract between the back end (analytics pipeline) and the front end (Dash dashboard).

Author: Saamarth Attray

This file is the single source of truth for:
  * where every pipeline output lives on disk
  * which columns the cleaned dataset must contain
  * the ordering / label sets the dashboard uses for categorical features
  * the top-level structure of model_results.json

It has NO third-party imports on purpose, so the dashboard (which does not install scikit-learn
or scipy) and the pipeline can both import it.

Zareena (back end): you own backend/pipeline.py. As long as the files below are written with the
documented columns / keys, the dashboard picks them up with no front-end changes.
Run `python -m backend.validate_outputs` after every pipeline run to check the contract.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------------------------------------
# File locations
# ----------------------------------------------------------------------------------------------
RAW_CSV = ROOT / "data" / "raw" / "faa_wildlife_strikes_2000_2025.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_PARQUET = PROCESSED_DIR / "strikes_clean.parquet"          # row level, cleaned + engineered
PROJECTION_PARQUET = PROCESSED_DIR / "projection_sample.parquet"  # <= 20k rows: PCA / LDA / cluster
RESULTS_JSON = PROCESSED_DIR / "model_results.json"               # everything else

# ----------------------------------------------------------------------------------------------
# Features (from the Phase 1 proposal, section 3.1)
# ----------------------------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "HEIGHT", "SPEED", "DISTANCE", "NUM_ENGS", "AC_MASS",
    "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ", "NUM_STRUCK",
    "INCIDENT_MONTH", "INCIDENT_YEAR",
]
CATEGORICAL_FEATURES = [
    "TIME_OF_DAY", "STATE", "OPERATOR", "AIRCRAFT", "TYPE_ENG",
    "PHASE_OF_FLIGHT", "SKY", "PRECIPITATION", "DAMAGE_LEVEL", "SPECIES",
]
# Engineered features (proposal, "Feature engineering")
ENGINEERED_FEATURES = ["STRIKE_CAUSED_DAMAGE", "SPECIES_GROUP", "SEASON", "TOTAL_COST"]

# ----------------------------------------------------------------------------------------------
# strikes_clean.parquet  -- columns the dashboard reads. Extra columns are fine, these are required.
# ----------------------------------------------------------------------------------------------
CLEAN_REQUIRED_COLUMNS = {
    "INDEX_NR": "int",
    "INCIDENT_YEAR": "int            2000-2025",
    "INCIDENT_MONTH": "int           1-12",
    "SEASON": "str                   Winter / Spring / Summer / Fall",
    "STATE": "str                    2-letter postal code",
    "AIRPORT": "str",
    "OPERATOR": "str",
    "AIRCRAFT": "str                 make/model",
    "TYPE_ENG": "str                 DECODED label, e.g. 'Turbofan' (raw file has letter codes A-F, Y)",
    "PHASE_OF_FLIGHT": "str          NaN allowed",
    "TIME_OF_DAY": "str              Dawn / Day / Dusk / Night, NaN allowed",
    "SKY": "str                      NaN allowed",
    "DAMAGE_LEVEL": "str             DECODED: None / Minor / Minor (uncertain) / Substantial / Destroyed; NaN = unknown",
    "SPECIES": "str",
    "SPECIES_GROUP": "str            engineered, see SPECIES_GROUPS",
    "STRIKE_CAUSED_DAMAGE": "float   1.0 / 0.0, NaN where DAMAGE_LEVEL is unknown (do NOT fill with 0)",
    "HEIGHT": "float                 ft AGL, NaN allowed",
    "SPEED": "float                  knots, NaN allowed",
    "DISTANCE": "float               nautical miles, NaN allowed",
    "NUM_ENGS": "float",
    "AC_MASS": "float                class 1-5",
    "NUM_STRUCK": "float             estimated animals struck (range midpoints), see build_dataset.py",
    "COST_REPAIRS_INFL_ADJ": "float  USD, NaN allowed (about 98% missing)",
    "COST_OTHER_INFL_ADJ": "float    USD, NaN allowed (about 98% missing)",
    "TOTAL_COST": "float             repairs + other, NaN only if BOTH are NaN",
}

# ----------------------------------------------------------------------------------------------
# Category labels / orderings used by the dashboard
# ----------------------------------------------------------------------------------------------
DAMAGE_LEVEL_ORDER = ["None", "Minor", "Minor (uncertain)", "Substantial", "Destroyed"]
TIME_OF_DAY_ORDER = ["Dawn", "Day", "Dusk", "Night"]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
PHASE_ORDER = [
    "Parked", "Taxi", "Take-off Run", "Departure", "Climb", "En Route",
    "Local", "Descent", "Approach", "Arrival", "Landing Roll",
]
SPECIES_GROUPS = [
    "Unknown bird", "Songbirds & perching", "Doves & pigeons", "Raptors", "Gulls & terns",
    "Waterfowl", "Shorebirds & waders", "Other birds", "Bats", "Other mammals", "Reptiles & other",
]
ENGINE_TYPE_LABELS = {
    "A": "Reciprocating", "B": "Turbojet", "C": "Turboprop", "D": "Turbofan",
    "E": "None (glider)", "F": "Turboshaft", "Y": "Other",
}

# ----------------------------------------------------------------------------------------------
# model_results.json  -- required top-level keys (see docs/DATA_CONTRACT.md for every field)
# ----------------------------------------------------------------------------------------------
RESULTS_REQUIRED_KEYS = [
    "meta", "classification", "pca", "lda", "clustering", "randomization", "normality", "outliers",
]

PROJECTION_REQUIRED_COLUMNS = ["INDEX_NR", "PC1", "PC2", "CLUSTER", "STRIKE_CAUSED_DAMAGE"]
