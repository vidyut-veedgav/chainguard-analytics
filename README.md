# ChurnGuard Analytics

XGBoost churn classifier over three CSV tables, surfaced through a Streamlit dashboard with a Gemini chat assistant.

See `docs/overview.md` for the brief, `docs/data_dictionary.md` for fields, and `docs/*_guide.md` for per-module details.

## Prerequisites

- **Python 3.13** (pinned in `.python-version`)
- **`uv`** (https://docs.astral.sh/uv/) — there is no `requirements.txt`
- Source CSVs in `data/raw/` (gitignored — get from the engagement drop):
  - `account_lifecycle_events.csv`
  - `user_engagement_metrics.csv`
  - `support_interaction_history.csv`
- Gemini credentials *(chat assistant only — dashboard works without)*

## Install

```bash
uv sync
```

## Configure Gemini auth

`src/agent.py` loads `.env` on import and supports two paths. Vertex wins when both are set.

```dotenv
# .env

# Path A — Vertex (uses gcloud Application Default Credentials)
VERTEX_PROJECT="your-gcp-project-id"
VERTEX_LOCATION="us-central1"

# Path B — Direct Gemini API (no GCP project)
GEMINI_API_KEY="..."
```

For Vertex: `gcloud auth application-default login` and enable the Vertex AI API on the project. For the direct path: get a key at https://aistudio.google.com/apikey and leave `VERTEX_PROJECT` unset.

Tunables live in `src/config.py` (snapshot date, leakage drops, XGBoost hyperparams, F-beta, threshold grid, artifact paths).

## Run the dashboard

```bash
uv run streamlit run app/dashboard.py --server.address 0.0.0.0
```

**Open the Network URL Streamlit prints, not `localhost`.** First-call latency is ~2–3s while the model loads and SHAP runs; everything after is cached.

## Retrain

Training is driven by `notebooks/eda.ipynb`, which calls `src/preprocessing.py` → `src/features.py` → `src/predict.py` and overwrites `models/{xgb_churn.json, config.json, feature_columns.json}`.

```bash
uv run jupyter lab notebooks/eda.ipynb  # run all cells
```

`notebooks/test_api.ipynb` smoke-tests `src/api.py` against the saved artifacts.

## Layout

```
data/raw/        # Source CSVs (gitignored)
docs/            # Brief, data dictionary, per-module guides
models/          # Persisted artifacts (committed)
notebooks/       # eda.ipynb (training), test_api.ipynb (smoke test)
src/
├── config.py         # All tunables
├── preprocessing.py  # CSV → merged feature frame
├── features.py       # Feature selection
├── predict.py        # Train, threshold sweep, save
├── api.py            # Cached scoring + SHAP for UI
├── agent.py          # Gemini function-calling layer
└── _validation.py    # Input guards
app/dashboard.py      # Streamlit app
```

## Troubleshooting

- `FileNotFoundError: data/raw/...csv` — CSVs not in place; see Prerequisites.
- `RuntimeError: No Gemini credentials found` — set `VERTEX_PROJECT` or `GEMINI_API_KEY`.
- `DefaultCredentialsError` — Vertex selected but ADC missing; run `gcloud auth application-default login` or unset `VERTEX_PROJECT`.
- Browser hangs on `localhost:8501` — use the Network URL.
