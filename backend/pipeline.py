"""
Baseline back-end analytics pipeline for the FAA Wildlife Strike dashboard.

Author: Saamarth Attray
(Written so the dashboard could be built and tested in parallel. Zareena owns the back end: replace or
extend any step. The dashboard only depends on the files / keys in backend/contract.py.)

Run:
    python -m backend.pipeline                # full run
    python -m backend.pipeline --skip-models  # cleaning + features only (fast, ~15 s)

Outputs (data/processed/):
    strikes_clean.parquet      row-level cleaned + engineered data
    projection_sample.parquet  PCA / LDA / cluster coordinates for <= 20,000 complete-case rows
    model_results.json         classification, PCA, LDA, clustering, randomization, normality, outliers

Steps (mirrors proposal section 3.3):
    1. load + clean            2. outlier detection        3. transformation
    4. normality testing       5. PCA + LDA                6. K-means clustering + isolation forest
    7. classification          8. randomization (permutation) tests
"""
import argparse
import json
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backend.contract import (CLEAN_PARQUET, ENGINE_TYPE_LABELS, PROCESSED_DIR, PROJECTION_PARQUET,
                              RAW_CSV, RESULTS_JSON, SPECIES_GROUPS)

SEED = 42
PIPELINE_VERSION = "0.1-baseline"

# ---- decisions Zareena may want to revisit ---------------------------------------------------
UNCERTAIN_COUNTS_AS_DAMAGE = True   # FAA code "M?" = minor damage, uncertain. True -> counts as damage
PCA_FEATURES = ["HEIGHT", "SPEED", "DISTANCE", "NUM_ENGS", "AC_MASS", "NUM_STRUCK"]
LOG_FEATURES = ["HEIGHT", "DISTANCE", "NUM_STRUCK", "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ"]
OUTLIER_Z = 3.0
PROJECTION_SAMPLE = 20_000
N_PERMUTATIONS = 5_000
# COST columns are only filled in when damage happened, so they must NOT be classifier inputs (target leakage).

DAMAGE_DECODE = {"N": "None", "M": "Minor", "M?": "Minor (uncertain)", "S": "Substantial", "D": "Destroyed"}

# ----------------------------------------------------------------------------------------------
# 1. Cleaning + feature engineering
# ----------------------------------------------------------------------------------------------
# Order matters: first matching rule wins.
_SPECIES_RULES = [
    ("Unknown bird", r"^unknown bird"),
    ("Reptiles & other", r"snake|turtle|terrapin|tortoise|lizard|iguana|alligator|crocodile|reptile|frog|toad"),
    ("Bats", r"bats?\b|pipistrelle"),
    # special cases whose names contain a mammal / raptor keyword
    ("Songbirds & perching", r"fox sparrow"),
    ("Shorebirds & waders", r"cattle egret"),
    ("Other birds", r"nighthawk|nightjar|poorwill|whip-poor-will"),
    ("Other mammals", r"coyote|skunk|\bdeer\b|\bfox(?:es)?\b|rabbit|hare\b|cottontail|woodchuck|groundhog|raccoon|"
                      r"opossum|prairie dog|\bdog\b|\bcat\b|squirrel|rodent|\bmouse\b|\brat\b|\belk\b|\bbear\b|"
                      r"bison|\bcow\b|horse|\bpig\b|armadillo|beaver|muskrat|nutria|bobcat|mammal|antelope|"
                      r"pronghorn|moose|goat|sheep|\bvole\b|\bmole\b|shrew|mongoose|weasel|badger|otter|"
                      r"porcupine|chipmunk|gopher|livestock"),
    ("Raptors", r"hawk|eagle|falcon|kestrel|merlin|osprey|\bkite|harrier|vulture|caracara|\bowl|raptor|"
                r"buzzard|condor"),
    ("Gulls & terns", r"\bgull|\bterns?\b|skimmer|jaeger|kittiwake|noddy"),
    ("Waterfowl", r"duck|goose|geese|swan|\bteal\b|mallard|pintail|wigeon|scaup|gadwall|shoveler|merganser|"
                  r"bufflehead|goldeneye|canvasback|redhead|eider|scoter|\bcoot|grebe|\bloon|\bbrant"),
    ("Shorebirds & waders", r"plover|sandpiper|killdeer|curlew|snipe|woodcock|dunlin|sanderling|phalarope|"
                            r"godwit|turnstone|stilt|avocet|oystercatcher|willet|yellowlegs|dowitcher|\bknot\b|"
                            r"lapwing|heron|egret|\bibis|bittern|\bcranes?\b|\brail\b|\bsora\b|stork|spoonbill|"
                            r"shorebird|wader|limpkin|gallinule|moorhen"),
    ("Doves & pigeons", r"dove|pigeon"),
    ("Songbirds & perching", r"swallow|sparrow|lark|starling|robin|finch|blackbird|grackle|cowbird|oriole|warbler|"
                             r"wren|thrush|bunting|junco|pipit|waxwing|kingbird|flycatcher|phoebe|pewee|vireo|"
                             r"kinglet|bluebird|mockingbird|catbird|thrasher|tanager|grosbeak|cardinal|crow|raven|"
                             r"\bjay|magpie|shrike|bobolink|dickcissel|longspur|siskin|redstart|ovenbird|"
                             r"yellowthroat|waterthrush|\bchat\b|veery|gnatcatcher|martin|myna|munia|titmouse|"
                             r"chickadee|nuthatch|creeper|bushtit|perching|passerine|euphonia|towhee|bulbul|"
                             r"weaver|whydah|waxbill|silverbill|parula"),
    ("Other birds", r"swift|hummingbird|woodpecker|sapsucker|flicker|cuckoo|pheasant|turkey|quail|grouse|"
                    r"partridge|kingfisher|cormorant|pelican|anhinga|gannet|booby|frigatebird|albatross|petrel|"
                    r"shearwater|puffin|murre|auk"),
]
_COMPILED_RULES = [(grp, re.compile(pat, re.IGNORECASE)) for grp, pat in _SPECIES_RULES]


def species_group(name) -> str:
    """Map a raw FAA species string to one of the broad groups in contract.SPECIES_GROUPS."""
    if not isinstance(name, str) or not name.strip():
        return "Unknown bird"
    for group, rx in _COMPILED_RULES:
        if rx.search(name):
            return group
    return "Other birds"


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["INCIDENT_DATE"] = pd.to_datetime(df["INCIDENT_DATE"], errors="coerce")
    for col in ["STATE", "AIRPORT", "AIRPORT_ID", "OPERATOR", "AIRCRAFT", "SPECIES",
                "TIME_OF_DAY", "PHASE_OF_FLIGHT", "SKY", "PRECIPITATION"]:
        df[col] = df[col].astype("string").str.strip()
        df.loc[df[col].isin(["", "Unknown", "UNKNOWN"]), col] = pd.NA
    df["STATE"] = df["STATE"].str.upper()

    # decode FAA letter codes into labels
    df["TYPE_ENG"] = df["TYPE_ENG"].astype("string").str.strip().map(ENGINE_TYPE_LABELS)
    df["DAMAGE_LEVEL"] = df["DAMAGE_LEVEL"].astype("string").str.strip().map(DAMAGE_DECODE)

    # engineered features (proposal section 3.1)
    damage_labels = ["Minor", "Substantial", "Destroyed"] + (["Minor (uncertain)"] if UNCERTAIN_COUNTS_AS_DAMAGE else [])
    known = df["DAMAGE_LEVEL"].notna()
    df["STRIKE_CAUSED_DAMAGE"] = np.where(df["DAMAGE_LEVEL"].isin(damage_labels), 1.0, 0.0)
    df.loc[~known, "STRIKE_CAUSED_DAMAGE"] = np.nan          # unknown stays unknown, never 0
    if not UNCERTAIN_COUNTS_AS_DAMAGE:
        df.loc[df["DAMAGE_LEVEL"] == "Minor (uncertain)", "STRIKE_CAUSED_DAMAGE"] = np.nan
    df["SPECIES_GROUP"] = df["SPECIES"].map(species_group)
    season = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
    df["SEASON"] = df["INCIDENT_MONTH"].map(season)
    df["TOTAL_COST"] = df[["COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ"]].sum(axis=1, min_count=1)

    # physically impossible values -> NaN (kept simple; extend as you find more)
    df.loc[df["HEIGHT"] < 0, "HEIGHT"] = np.nan
    df.loc[df["SPEED"] <= 0, "SPEED"] = np.nan
    df.loc[df["DISTANCE"] < 0, "DISTANCE"] = np.nan
    return df


# ----------------------------------------------------------------------------------------------
# 2. Outliers
# ----------------------------------------------------------------------------------------------
def _log(series: pd.Series, name: str) -> pd.Series:
    return np.log1p(series) if name in LOG_FEATURES else series


def detect_outliers(df: pd.DataFrame):
    """IQR (1.5x) and |z| > 3 on each skewed-aware feature. Flags rows, never deletes them."""
    report = {"methods": ["IQR 1.5x fence", f"|z| > {OUTLIER_Z}"],
              "log_scale_features": LOG_FEATURES, "features": {}}
    z_flags = pd.DataFrame(index=df.index)
    for col in ["HEIGHT", "SPEED", "DISTANCE", "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ"]:
        s = _log(df[col], col)
        valid = s.dropna()
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        z = (s - valid.mean()) / valid.std()
        iqr_flag = (s < lo) | (s > hi)
        z_flag = z.abs() > OUTLIER_Z
        z_flags[col] = z_flag.fillna(False)
        back = (lambda v: float(np.expm1(v))) if col in LOG_FEATURES else float
        report["features"][col] = {
            "n_valid": int(valid.size),
            "iqr_outliers": int(iqr_flag.sum()), "z_outliers": int(z_flag.sum()),
            "iqr_lower_bound": back(lo), "iqr_upper_bound": back(hi),
            "max_value": float(df[col].max()),
        }
    df["OUTLIER_HEIGHT"] = z_flags["HEIGHT"]
    df["OUTLIER_SPEED"] = z_flags["SPEED"]
    df["OUTLIER_DISTANCE"] = z_flags["DISTANCE"]
    df["OUTLIER_COST"] = z_flags["COST_REPAIRS_INFL_ADJ"] | z_flags["COST_OTHER_INFL_ADJ"]
    df["OUTLIER_ANY"] = df[["OUTLIER_HEIGHT", "OUTLIER_SPEED", "OUTLIER_DISTANCE", "OUTLIER_COST"]].any(axis=1)
    report["rows_flagged_any"] = int(df["OUTLIER_ANY"].sum())
    report["policy"] = ("Rows are flagged, not deleted. HEIGHT/SPEED/DISTANCE z-outliers are excluded from PCA and "
                        "clustering only. Cost outliers are real high-severity events and are kept.")
    top = df.nlargest(10, "TOTAL_COST")[["INCIDENT_YEAR", "STATE", "AIRCRAFT", "SPECIES", "TOTAL_COST"]]
    report["top_cost_events"] = [
        {"year": int(r.INCIDENT_YEAR), "state": None if pd.isna(r.STATE) else str(r.STATE),
         "aircraft": None if pd.isna(r.AIRCRAFT) else str(r.AIRCRAFT),
         "species": None if pd.isna(r.SPECIES) else str(r.SPECIES), "total_cost": float(r.TOTAL_COST)}
        for r in top.itertuples()]
    return df, report


# ----------------------------------------------------------------------------------------------
# 3-4. Transformation + normality
# ----------------------------------------------------------------------------------------------
def _qq(values: np.ndarray, n_points: int = 100):
    probs = (np.arange(1, n_points + 1) - 0.5) / n_points
    return {"theoretical": [float(x) for x in stats.norm.ppf(probs)],
            "sample": [float(x) for x in np.quantile(values, probs)]}


def _normality_for(values: np.ndarray, rng) -> dict:
    sample = rng.choice(values, size=min(5000, values.size), replace=False)   # Shapiro limit
    sw = stats.shapiro(sample)
    k2 = stats.normaltest(values)
    ad = stats.anderson(values, dist="norm", method="interpolate")
    return {
        "n": int(values.size), "skew": float(stats.skew(values)), "kurtosis": float(stats.kurtosis(values)),
        "shapiro_W": float(sw.statistic), "shapiro_p": float(sw.pvalue),
        "dagostino_K2": float(k2.statistic), "dagostino_p": float(k2.pvalue),
        "anderson_A2": float(ad.statistic), "anderson_p": float(ad.pvalue),
        "normal_by_all_three": bool(sw.pvalue > 0.05 and k2.pvalue > 0.05 and ad.pvalue > 0.05),
        "qq": _qq(values),
    }


def run_normality(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    feats = ["HEIGHT", "SPEED", "DISTANCE", "NUM_ENGS", "AC_MASS", "NUM_STRUCK",
             "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ"]
    out = {"methods": ["Shapiro-Wilk (5,000-row sample)", "D'Agostino-Pearson K2", "Anderson-Darling"],
           "alpha": 0.05, "features": {}}
    for col in feats:
        s = df[col].dropna().to_numpy(dtype=float)
        if s.size < 20:
            continue
        entry = {"raw": _normality_for(s, rng)}
        if col in LOG_FEATURES:
            entry["log1p"] = _normality_for(np.log1p(s), rng)
        out["features"][col] = entry
    out["note"] = ("With n in the thousands, even trivial departures reject normality; read the Q-Q plots and skew "
                   "alongside the p-values.")
    return out


# ----------------------------------------------------------------------------------------------
# 5-6. PCA, LDA, clustering, isolation forest
# ----------------------------------------------------------------------------------------------
def _transform_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[PCA_FEATURES].copy()
    for col in PCA_FEATURES:
        if col in LOG_FEATURES:
            X[col] = np.log1p(X[col])
    return X


def run_dimensionality_and_clusters(df: pd.DataFrame):
    complete = df.dropna(subset=PCA_FEATURES)
    complete = complete[~(complete["OUTLIER_HEIGHT"] | complete["OUTLIER_SPEED"] | complete["OUTLIER_DISTANCE"])]
    X = _transform_matrix(complete)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # PCA
    pca = PCA(n_components=len(PCA_FEATURES), random_state=SEED).fit(Xs)
    scores = pca.transform(Xs)
    pca_out = {
        "features": PCA_FEATURES, "n_rows": int(len(complete)),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "cumulative_variance": [float(v) for v in np.cumsum(pca.explained_variance_ratio_)],
        "loadings": {f: [float(v) for v in pca.components_[:, i]] for i, f in enumerate(PCA_FEATURES)},
        "note": "Complete cases only (HEIGHT, SPEED, DISTANCE all reported), z-outliers removed, log1p on skewed "
                "features, then standardized.",
    }

    # K-means: choose k by silhouette on a sample
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xs), size=min(10_000, len(Xs)), replace=False)
    k_scan = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(Xs)
        k_scan.append({"k": k, "inertia": float(km.inertia_),
                       "silhouette": float(silhouette_score(Xs[idx], km.labels_[idx]))})
    best_k = max(k_scan, key=lambda d: d["silhouette"])["k"]
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Xs)
    labels = km.labels_
    profiles = []
    for c in range(best_k):
        sub = complete[labels == c]
        known = sub["STRIKE_CAUSED_DAMAGE"].dropna()
        profiles.append({
            "cluster": f"Cluster {c + 1}", "n": int(len(sub)), "share": float(len(sub) / len(complete)),
            "damage_rate": float(known.mean()) if len(known) else None,
            "top_phase": None if sub["PHASE_OF_FLIGHT"].dropna().empty else str(sub["PHASE_OF_FLIGHT"].mode().iloc[0]),
            "top_species_group": str(sub["SPECIES_GROUP"].mode().iloc[0]),
            "means": {f: float(sub[f].mean()) for f in PCA_FEATURES},
        })
    cluster_out = {"algorithm": "K-means", "k": int(best_k), "selection": "highest silhouette for k = 2..8",
                   "k_scan": k_scan, "silhouette": float(next(d["silhouette"] for d in k_scan if d["k"] == best_k)),
                   "profiles": profiles}

    # Isolation forest (multivariate outliers)
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=SEED, n_jobs=-1).fit(Xs)
    iso_flag = iso.predict(Xs) == -1

    # LDA (binary target -> 1 discriminant axis) on rows with a known outcome
    y = complete["STRIKE_CAUSED_DAMAGE"]
    known_mask = y.notna().to_numpy()
    lda = LinearDiscriminantAnalysis(n_components=1).fit(Xs[known_mask], y[known_mask].astype(int))
    ld1 = np.full(len(Xs), np.nan)
    ld1[known_mask] = lda.transform(Xs[known_mask])[:, 0]
    lda_out = {
        "n_rows": int(known_mask.sum()), "n_components": 1,
        "explained_variance_ratio": [1.0],
        "coefficients": {f: float(v) for f, v in zip(PCA_FEATURES, lda.scalings_[:, 0])},
        "training_accuracy": float(lda.score(Xs[known_mask], y[known_mask].astype(int))),
        "majority_class_baseline": float(max(y[known_mask].mean(), 1 - y[known_mask].mean())),
        "class_means_ld1": {"No damage": float(np.nanmean(ld1[known_mask][y[known_mask].to_numpy() == 0])),
                            "Damage": float(np.nanmean(ld1[known_mask][y[known_mask].to_numpy() == 1]))},
        "note": "Two classes -> one discriminant axis (column LD1 in projection_sample.parquet). Compare "
                "training_accuracy with majority_class_baseline: with ~9% damage, 'always no damage' already scores ~91%.",
    }

    proj = pd.DataFrame({
        "INDEX_NR": complete["INDEX_NR"].to_numpy(),
        "PC1": scores[:, 0], "PC2": scores[:, 1], "PC3": scores[:, 2],
        "CLUSTER": [f"Cluster {c + 1}" for c in labels],
        "STRIKE_CAUSED_DAMAGE": complete["STRIKE_CAUSED_DAMAGE"].to_numpy(),
        "LD1": ld1, "ISO_OUTLIER": iso_flag,
        "SPECIES_GROUP": complete["SPECIES_GROUP"].to_numpy(),
        "PHASE_OF_FLIGHT": complete["PHASE_OF_FLIGHT"].to_numpy(),
    })
    proj = proj.sample(n=min(PROJECTION_SAMPLE, len(proj)), random_state=SEED).reset_index(drop=True)
    outlier_extra = {"isolation_forest": {"contamination": 0.01, "n_flagged": int(iso_flag.sum()),
                                          "n_rows": int(len(Xs))}}
    return pca_out, lda_out, cluster_out, proj, outlier_extra


# ----------------------------------------------------------------------------------------------
# 7. Classification
# ----------------------------------------------------------------------------------------------
CLF_NUMERIC = ["HEIGHT", "SPEED", "DISTANCE", "NUM_ENGS", "AC_MASS", "NUM_STRUCK", "INCIDENT_MONTH", "INCIDENT_YEAR"]
CLF_MISSING_FLAGS = ["HEIGHT", "SPEED", "DISTANCE"]
CLF_CATEGORICAL = ["PHASE_OF_FLIGHT", "SPECIES_GROUP", "TIME_OF_DAY", "SKY", "TYPE_ENG"]


def _design_matrix(frame: pd.DataFrame, medians=None, columns=None):
    X = frame[CLF_NUMERIC].copy()
    for col in CLF_MISSING_FLAGS:
        X[f"{col}_MISSING"] = frame[col].isna().astype(int)
    for col in LOG_FEATURES:
        if col in X:
            X[col] = np.log1p(X[col])
    medians = X.median() if medians is None else medians
    X = X.fillna(medians)
    cats = frame[CLF_CATEGORICAL].astype("object").where(frame[CLF_CATEGORICAL].notna(), "Unknown")
    X = pd.concat([X, pd.get_dummies(cats, prefix=CLF_CATEGORICAL, dtype=int)], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X, medians


def _group_of(col: str) -> str:
    for c in CLF_CATEGORICAL:
        if col.startswith(c + "_"):
            return c
    return col.replace("_MISSING", " (missing flag)")


def run_classification(df: pd.DataFrame) -> dict:
    known = df.dropna(subset=["STRIKE_CAUSED_DAMAGE"])
    X, medians = _design_matrix(known)
    y = known["STRIKE_CAUSED_DAMAGE"].astype(int).to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    scaler = StandardScaler().fit(X_tr)
    models = {
        "Logistic regression": (LogisticRegression(max_iter=1000, class_weight="balanced"), True),
        "Random forest": (RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=20,
                                                 class_weight="balanced_subsample", n_jobs=-1,
                                                 random_state=SEED), False),
    }
    results, importances = [], {}
    for name, (model, needs_scale) in models.items():
        Xa, Xb = (scaler.transform(X_tr), scaler.transform(X_te)) if needs_scale else (X_tr, X_te)
        model.fit(Xa, y_tr)
        proba = model.predict_proba(Xb)[:, 1]
        pred = (proba >= 0.5).astype(int)
        cm = confusion_matrix(y_te, pred, labels=[0, 1])
        results.append({
            "name": name,
            "metrics": {"accuracy": float(accuracy_score(y_te, pred)),
                        "precision": float(precision_score(y_te, pred, zero_division=0)),
                        "recall": float(recall_score(y_te, pred)),
                        "f1": float(f1_score(y_te, pred)),
                        "roc_auc": float(roc_auc_score(y_te, proba)),
                        "pr_auc": float(average_precision_score(y_te, proba))},
            "confusion_matrix": {"labels": ["No damage", "Damage"], "matrix": cm.tolist()},
        })
        if name == "Random forest":
            agg = {}
            for col, imp in zip(X.columns, model.feature_importances_):
                agg[_group_of(col)] = agg.get(_group_of(col), 0.0) + float(imp)
            importances[name] = sorted(({"feature": k, "importance": v} for k, v in agg.items()),
                                       key=lambda d: -d["importance"])
    best = max(results, key=lambda r: r["metrics"]["roc_auc"])["name"]
    return {
        "target": "STRIKE_CAUSED_DAMAGE", "n_rows": int(len(known)), "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)), "positive_rate": float(y.mean()), "threshold": 0.5,
        "split": "80/20 stratified, random_state=42", "best_model": best, "models": results,
        "feature_importance": importances["Random forest"],
        "features_used": CLF_NUMERIC + [f"{c}_MISSING" for c in CLF_MISSING_FLAGS] + CLF_CATEGORICAL,
        "excluded_for_leakage": ["COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ", "TOTAL_COST", "DAMAGE_LEVEL"],
        "note": ("Rows with unknown damage level are excluded. Classes are imbalanced, so ROC-AUC and PR-AUC are "
                 "more informative than accuracy."),
    }


# ----------------------------------------------------------------------------------------------
# 8. Randomization (permutation) tests
# ----------------------------------------------------------------------------------------------
def _perm_test(df, name, question, mask_a, mask_b, label_a, label_b, rng) -> dict:
    sub = df.dropna(subset=["STRIKE_CAUSED_DAMAGE"])
    a = sub.loc[mask_a(sub), "STRIKE_CAUSED_DAMAGE"].to_numpy()
    b = sub.loc[mask_b(sub), "STRIKE_CAUSED_DAMAGE"].to_numpy()
    observed = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = len(a)
    null = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        rng.shuffle(pooled)
        null[i] = pooled[:n_a].mean() - pooled[n_a:].mean()
    p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (N_PERMUTATIONS + 1))
    return {
        "name": name, "question": question, "statistic": "difference in damage rate (A - B)",
        "group_a": {"label": label_a, "n": int(n_a), "damage_rate": float(a.mean())},
        "group_b": {"label": label_b, "n": int(len(b)), "damage_rate": float(b.mean())},
        "observed_diff": observed, "p_value": p, "n_permutations": N_PERMUTATIONS, "alternative": "two-sided",
        "null_distribution": [float(x) for x in null[:2000]],
    }


def run_randomization(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    tests = [
        _perm_test(df, "Day vs. night", "Is the damage rate different for strikes at night?",
                   lambda s: s["TIME_OF_DAY"] == "Night", lambda s: s["TIME_OF_DAY"] == "Day",
                   "Night", "Day", rng),
        _perm_test(df, "Approach vs. take-off run", "Are approach strikes more or less damaging than take-off-run strikes?",
                   lambda s: s["PHASE_OF_FLIGHT"] == "Approach", lambda s: s["PHASE_OF_FLIGHT"] == "Take-off Run",
                   "Approach", "Take-off Run", rng),
    ]
    return {"method": "Permutation test: shuffle group labels, recompute the difference in damage rate",
            "tests": tests}


# ----------------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def _sanitize(obj):
    """Replace NaN / inf floats with None so the JSON is strictly valid."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main(skip_models: bool = False) -> None:
    t0 = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("1/8 loading + cleaning ...")
    df = clean(load_raw())
    print("2/8 outliers ...")
    df, outliers = detect_outliers(df)

    clean_cols = [
        "INDEX_NR", "INCIDENT_DATE", "INCIDENT_YEAR", "INCIDENT_MONTH", "SEASON", "STATE", "AIRPORT", "AIRPORT_ID",
        "OPERATOR", "AIRCRAFT", "TYPE_ENG", "PHASE_OF_FLIGHT", "TIME_OF_DAY", "SKY", "PRECIPITATION",
        "DAMAGE_LEVEL", "SPECIES", "SPECIES_GROUP", "STRIKE_CAUSED_DAMAGE",
        "HEIGHT", "SPEED", "DISTANCE", "NUM_ENGS", "AC_MASS", "NUM_STRUCK",
        "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ", "TOTAL_COST",
        "OUTLIER_HEIGHT", "OUTLIER_SPEED", "OUTLIER_DISTANCE", "OUTLIER_COST", "OUTLIER_ANY",
    ]
    out = df[clean_cols].copy()
    for col in ["STATE", "AIRPORT", "AIRPORT_ID", "OPERATOR", "AIRCRAFT", "TYPE_ENG", "PHASE_OF_FLIGHT",
                "TIME_OF_DAY", "SKY", "PRECIPITATION", "DAMAGE_LEVEL", "SPECIES", "SPECIES_GROUP", "SEASON"]:
        out[col] = out[col].astype("object").where(out[col].notna(), None)
    out.to_parquet(CLEAN_PARQUET, index=False)
    print(f"   wrote {CLEAN_PARQUET.name}: {len(out):,} rows")

    known = df["STRIKE_CAUSED_DAMAGE"].dropna()
    meta = {
        "pipeline_version": PIPELINE_VERSION, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(df)), "n_known_damage": int(len(known)), "damage_rate_known": float(known.mean()),
        "year_min": int(df["INCIDENT_YEAR"].min()), "year_max": int(df["INCIDENT_YEAR"].max()),
        "uncertain_counts_as_damage": UNCERTAIN_COUNTS_AS_DAMAGE,
        "species_groups": SPECIES_GROUPS,
        "missing_pct": {c: float(df[c].isna().mean() * 100) for c in
                        ["HEIGHT", "SPEED", "DISTANCE", "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ",
                         "DAMAGE_LEVEL", "PHASE_OF_FLIGHT", "TIME_OF_DAY"]},
    }
    if skip_models:
        print("   --skip-models set: not writing model_results.json")
        return

    print("3/8 normality ...")
    normality = run_normality(df)
    print("4-6/8 PCA, LDA, K-means, isolation forest ...")
    pca_out, lda_out, cluster_out, proj, outlier_extra = run_dimensionality_and_clusters(df)
    outliers.update(outlier_extra)
    proj.to_parquet(PROJECTION_PARQUET, index=False)
    print(f"   wrote {PROJECTION_PARQUET.name}: {len(proj):,} rows")
    print("7/8 classification ...")
    classification = run_classification(df)
    print("8/8 permutation tests ...")
    randomization = run_randomization(df)

    results = _sanitize({"meta": meta, "classification": classification, "pca": pca_out, "lda": lda_out,
                         "clustering": cluster_out, "randomization": randomization,
                         "normality": normality, "outliers": outliers})
    RESULTS_JSON.write_text(json.dumps(results, default=_json_default))
    print(f"   wrote {RESULTS_JSON.name} ({RESULTS_JSON.stat().st_size / 1e6:.2f} MB)")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-models", action="store_true", help="only write strikes_clean.parquet")
    main(ap.parse_args().skip_models)
