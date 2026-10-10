# Wildlife strikes on aircraft: CS/STAT 3654 term project

FAA National Wildlife Strike Database (2000-2024; 2025 has only 16 rows).
Front end: Saamarth Attray (Dash dashboard, Docker, GCP). Back end: Zareena Yussouff (pipeline and models).

## Run it
```bash
pip install -r requirements.txt
python app.py            # http://127.0.0.1:8050
```
The dashboard reads three files from `data/processed/`. If one is missing, only the part that needs it shows a notice.

## Rebuild the data and models (Zareena)
```bash
pip install -r requirements-backend.txt
PYTHONPATH=. python3 -m backend.build_dataset path/to/faa_aws.csv   # writes data/raw/...csv
PYTHONPATH=. python3 -m backend.pipeline                            # ~35 s; --skip-models for cleaning only
PYTHONPATH=. python3 -m backend.validate_outputs                    # checks the file contract
```
The exact columns and JSON keys are in `docs/DATA_CONTRACT.md` and `backend/contract.py`.

## Docker
```bash
docker build -t faa-dashboard .
docker run -p 8080:8080 faa-dashboard      # http://localhost:8080
```

## Deploy to Google Cloud Run
```bash
gcloud auth login
PROJECT_ID=your-project ./deploy/deploy_cloud_run.sh   # prints the public URL
```

## Data notes
- ~37% of damage outcomes are unknown; they are never treated as "no damage".
- HEIGHT is 52% missing, SPEED 70%, COST ~98%. Cost columns are left out of the classifier (they leak the answer).
- The share of strikes with a reported outcome fell from ~80% to ~60% over time.
