# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

ChurnGuard Analytics is a consulting proof-of-concept that predicts SaaS account churn from three CSV tables and exposes the results through a chat + dashboard interface. Full brief in `docs/overview.md`; field-level reference in `docs/data_dictionary.md`. Read both before doing data work — they document data-quality issues you cannot infer from the CSVs.

The engagement has two phases: Phase 1 builds the system; Phase 2 is a live session where the brief explicitly warns a "curveball" will require adapting the system. **Architect for flexibility** — don't hard-code data source paths, schemas, or feature lists. Keep data loading, feature engineering, modeling, and UI cleanly separated so any one layer can be swapped quickly.

## Tooling

- Python **3.13** (`.python-version`), managed with **`uv`** — there is no `requirements.txt`. Use `uv sync` to install, `uv run <cmd>` to execute, `uv add <pkg>` to add deps (updates `pyproject.toml` + `uv.lock`).
- Core stack: `pandas`, `numpy`, `scikit-learn`, `sweetviz` (used for the HTMLs in `data/profiles/`), `ipykernel` for notebooks.

## Layout

```
data/raw/           # Source CSVs (gitignored): account_lifecycle_events, user_engagement_metrics, support_interaction_history
data/profiles/      # Sweetviz EDA reports (one per table)
docs/               # Brief + data dictionary — authoritative
notebooks/eda.ipynb # Exploratory work
src/                # (empty) intended for data loading, features, modeling
app/                # (empty) intended for chat + dashboard UI
main.py             # Placeholder entry point
```

`src/` and `app/` are currently empty `__init__.py` packages — the modeling and app code has not been written yet.

## Data gotchas (from the data dictionary — easy to miss)

- **Joins:** EU accounts post-Jan 2023 have a populated `account_uuid`; older/non-EU accounts have it NULL. Always join on `account_id` (present everywhere) — `account_uuid` is supplementary.
- **Target:** churn = `account_status == 'churned'` in `account_lifecycle_events`. `grace_period` was backfilled incompletely, so some "active" rows are really pre-churn.
- **Leakage risks to watch:** `account_health_score` (algorithm changed Q3 2023, behaves oddly in models), `risk_flag` (set by CS team, NULL for ~50% of accounts, only Pro/Enterprise tiers), `cancellation_requested` and `cancellation_reason` in support tickets, and any `status_change_date` near the prediction horizon.
- **Deprecated/duplicate fields:** `csat_score` duplicates `satisfaction_rating`; `internal_category` is being phased out; `org_hierarchy` table is referenced but does not exist.
- `user_id` in support tickets does not always match `user_engagement_metrics` (legacy migration gap).

## Conventions

- The brief explicitly says: skip Docker/K8s, skip extensive hyperparameter tuning, skip polished UI. Functional and well-structured beats clever.
- The chat interface should give *interpretable* explanations (feature contributions per account), not just probabilities.
- **Mutate dataframes in-place; never create renamed copies** (`df2`, `df_clean`, etc.). All feature engineering functions — column conversions, drops, derived columns — should add/drop directly on the existing dataframe. This keeps the namespace flat and avoids stale-copy bugs where a downstream cell still references the old frame.
- **Single exception — `pp_df` is the working frame.** Source frames (`ale_df`, `uem_df`, `sih_df`) mirror `data/raw` and are read-only after the load cell — no mutations, no drops, no derived columns. `pp_df` is born from the merge (`pp_df = ale_df.merge(uem_agg, ...).merge(sih_agg, ...)`) and is the single named copy thereafter. All preprocessing (drops, encoding, target binarization) and all analytical drops (variance, correlation, leakage, model-based selection) happen on `pp_df` in place. This keeps the source frames re-runnable without re-deriving features and makes every drop decision auditable in one place.
- **Decisions update the corresponding guide.** Every `docs/*_guide.md` file documents the *current* state of one slice of the system (`preprocessing_guide.md` → EDA decisions, `feature_selection_guide.md` → selection decisions, `features_module_guide.md` → `src/preprocessing.py` contract, etc.). Whenever a decision lands — adding a feature, dropping a leaky field, changing a snapshot date, renaming a module, removing a redundant constant — update the matching guide in the same change. The guides are how Phase 2 (and future-you) recover context quickly; stale guides are worse than missing ones.
