# Data contract: back end -> front end

Everything lives in `data/processed/`. `backend/contract.py` is the source of truth; run
`PYTHONPATH=. python3 -m backend.validate_outputs` after changing the pipeline.

| File | What it is |
|---|---|
| `strikes_clean.parquet` | One row per strike, cleaned. Required columns listed in `CLEAN_REQUIRED_COLUMNS`. |
| `projection_sample.parquet` | <= 20,000 rows: `INDEX_NR, PC1, PC2, CLUSTER, STRIKE_CAUSED_DAMAGE` (+ optional outlier flag). |
| `model_results.json` | Classification, PCA, LDA, clustering, randomization, normality and outlier results. Keys in `RESULTS_REQUIRED_KEYS`. |

Rules the dashboard relies on
- `STRIKE_CAUSED_DAMAGE` is 1.0 / 0.0 / NaN. NaN = outcome not reported (never 0).
- `DAMAGE_LEVEL` uses decoded labels: None, Minor, Minor (uncertain), Substantial, Destroyed.
- `SPECIES_GROUP` must be one of `SPECIES_GROUPS`.
- Add new keys freely; do not rename existing ones without telling Sam.
