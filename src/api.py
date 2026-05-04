"""API layer for the chat interface and dashboard.

Lazy-loaded module-level cache (LRU strategy).

Steps: 

    1. Load model weights, config, feature list
    2. Preprocess raw CSVs and reindex to selected features
    3. Score probabilities for every account
    4. Compute per-row Tree SHAP contributions (log-odds)

Subsequent calls have cache hits and return JSON objects

`_DATA_PATHS` can be changed before the first API call to point at alternate data sources.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

from src.config import (
    FEATURE_COLUMNS_PATH,
    MODEL_CONFIG_PATH,
    MODEL_WEIGHTS_PATH,
)
from src.features import apply_locked_features
from src.preprocessing import build_feature_frame

_DATA_PATHS = {
    'ale': Path('data/raw/account_lifecycle_events.csv'),
    'uem': Path('data/raw/user_engagement_metrics.csv'),
    'sih': Path('data/raw/support_interaction_history.csv'),
}

_PROB_COL = 'probability'
_PRED_COL = 'predicted'
_STATUS_COL = 'actual_status'
_BIAS_COL = '_bias'


# ---------------------------------------------------------------------------
# Caching strategy
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_model() -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(MODEL_WEIGHTS_PATH)
    return model


@lru_cache(maxsize=1)
def _load_config() -> dict:
    with open(MODEL_CONFIG_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_feature_columns() -> tuple[str, ...]:
    with open(FEATURE_COLUMNS_PATH) as f:
        return tuple(json.load(f))


@lru_cache(maxsize=1)
def _load_scoring_frame() -> pd.DataFrame:
    """Preprocessed + scored frame, indexed by account_id.

    `actual_status` is 'active' or 'churned' (the binarized target). Already-churned
    accounts are kept in the frame so per-account lookups still work, but the
    portfolio-level methods filter them out — they're not actionable risk.
    """
    pp = build_feature_frame(_DATA_PATHS['ale'], _DATA_PATHS['uem'], _DATA_PATHS['sih'])
    X, y = apply_locked_features(pp)
    model = _load_model()
    threshold = _load_config()['threshold']
    proba = model.predict_proba(X)[:, 1]
    out = X.copy()
    out[_PROB_COL] = proba
    out[_PRED_COL] = (proba >= threshold).astype(int)
    out[_STATUS_COL] = np.where(y.values == 1, 'churned', 'active')
    return out


def _active_subset(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame[_STATUS_COL] == 'active']


@lru_cache(maxsize=1)
def _load_pred_contribs() -> pd.DataFrame:
    """Per-row Tree SHAP contributions in log-odds space.

    Last column (_bias) is the global bias. Each row sums to the raw margin;
    expit(margin) == probability of class 1.
    """
    frame = _load_scoring_frame()
    feature_cols = list(_load_feature_columns())
    X = frame[feature_cols]
    booster = _load_model().get_booster()
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)  # (n, F+1), bias last
    return pd.DataFrame(contribs, index=X.index, columns=feature_cols + [_BIAS_COL])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _label(predicted: int) -> str:
    return 'churn' if predicted == 1 else 'retain'


def _row_or_raise(account_id: str) -> pd.Series:
    frame = _load_scoring_frame()
    if account_id not in frame.index:
        raise KeyError(f"account_id {account_id!r} not in scoring frame")
    return frame.loc[account_id]


def _scalarize(v):
    # numpy/pandas scalars -> python primitives so dicts json-serialize cleanly.
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_account_risk(account_id: str) -> dict:
    """Probability + predicted label for a single account.

    Works for both active and already-churned accounts. `actual_status` lets the
    caller frame the answer ("0.99 confirms a known churn" vs "0.99 is a live alert").
    """
    row = _row_or_raise(account_id)
    return {
        'account_id':      account_id,
        'probability':     float(row[_PROB_COL]),
        'predicted_label': _label(int(row[_PRED_COL])),
        'actual_status':   str(row[_STATUS_COL]),
        'threshold':       float(_load_config()['threshold']),
    }


def explain_account(account_id: str, top_k: int = 3) -> dict:
    """Per-account churn explanation via Tree SHAP (log-odds contributions).

    `top_drivers` are the top_k features pushing toward churn (positive
    contributions); `top_protectors` are the top_k pushing toward retention.
    Returning both lets the agent answer "why high" or "why low" symmetrically.
    """
    row = _row_or_raise(account_id)
    contribs = _load_pred_contribs().loc[account_id]
    feature_cols = list(_load_feature_columns())
    feature_values = row[feature_cols]
    feature_contribs = contribs.drop(_BIAS_COL).sort_values(ascending=False)

    def _entry(feat: str) -> dict:
        c = float(feature_contribs[feat])
        return {
            'feature':              feat,
            'value':                _scalarize(feature_values[feat]),
            'contribution_logodds': c,
            'direction':            'increases_risk' if c > 0 else 'decreases_risk',
        }

    positives = feature_contribs[feature_contribs > 0]
    negatives = feature_contribs[feature_contribs < 0]
    drivers    = [_entry(f) for f in positives.head(top_k).index]
    protectors = [_entry(f) for f in negatives.tail(top_k)[::-1].index]  # most negative first

    return {
        'account_id':      account_id,
        'probability':     float(row[_PROB_COL]),
        'predicted_label': _label(int(row[_PRED_COL])),
        'actual_status':   str(row[_STATUS_COL]),
        'threshold':       float(_load_config()['threshold']),
        'base_value':      float(contribs[_BIAS_COL]),
        'top_drivers':     drivers,
        'top_protectors':  protectors,
    }


def top_risk_accounts(n: int = 5) -> list[dict]:
    """N currently-active accounts with the highest predicted churn probability.

    Already-churned accounts are filtered out — they're memorized training labels,
    not actionable risk. Use `get_account_risk(id)` for per-account lookups
    regardless of status.
    """
    top = _active_subset(_load_scoring_frame()).nlargest(n, _PROB_COL)
    return [
        {
            'account_id':      str(idx),
            'probability':     float(row[_PROB_COL]),
            'predicted_label': _label(int(row[_PRED_COL])),
        }
        for idx, row in top.iterrows()
    ]


def top_k_accounts(k: int) -> dict:
    """Capacity-aware variant of `top_risk_accounts`.

    Returns the top K active accounts plus the `implied_threshold` — the predicted
    probability at position K. Companion to the fixed-threshold view in
    `portfolio_summary().n_high_risk`: pick whichever framing matches the binding
    constraint (capacity vs. risk cutoff). See `docs/api_module_guide.md` for the
    consumption pattern.

    `k` is capped at the size of the active book.
    """
    active = _active_subset(_load_scoring_frame())
    k = max(0, min(k, len(active)))
    top = active.nlargest(k, _PROB_COL) if k else active.iloc[:0]
    implied_threshold = float(top[_PROB_COL].iloc[-1]) if k else None
    return {
        'k':                 k,
        'implied_threshold': implied_threshold,
        'accounts': [
            {
                'account_id':      str(idx),
                'probability':     float(row[_PROB_COL]),
                'predicted_label': _label(int(row[_PRED_COL])),
            }
            for idx, row in top.iterrows()
        ],
    }


def feature_importance(top_k: int = 10) -> list[dict]:
    """Global feature importance from the XGBoost model (gain), descending."""
    model = _load_model()
    feature_cols = list(_load_feature_columns())
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False).head(top_k)
    return [
        {'rank': i + 1, 'feature': feat, 'importance': float(val)}
        for i, (feat, val) in enumerate(importances.items())
    ]


def portfolio_summary() -> dict:
    """Dashboard rollups: portfolio stats + held-out model metrics from config.json.

    `n_high_risk`, `pct_high_risk`, and `avg_probability` are computed across the
    ACTIVE book — already-churned accounts inflate these numbers without being
    actionable. `total_accounts` / `n_active` / `n_churned` give the full breakdown.

    Held-out metrics are NOT recomputed on the scoring frame; the deployed model
    was refit on all rows, so any in-sample metric would leak.
    """
    frame = _load_scoring_frame()
    active = _active_subset(frame)
    cfg = _load_config()
    threshold = float(cfg['threshold'])
    holdout = cfg['holdout_metrics']
    at_thr = holdout['at_threshold']

    total = int(len(frame))
    n_active = int(len(active))
    n_high = int((active[_PROB_COL] >= threshold).sum())

    return {
        'total_accounts':  total,
        'n_active':        n_active,
        'n_churned':       total - n_active,
        'n_high_risk':     n_high,
        'pct_high_risk':   float(n_high / n_active) if n_active else 0.0,
        'avg_probability': float(active[_PROB_COL].mean()) if n_active else 0.0,
        'threshold':       threshold,
        'model_metrics': {
            'precision':         at_thr['precision'],
            'recall':            at_thr['recall'],
            'f1':                at_thr['f1'],
            'f2':                at_thr['f2'],
            'average_precision': holdout['average_precision'],
            'n_test_rows':       holdout['n_test_rows'],
            'n_test_positives':  holdout['n_test_positives'],
            'trained_at':        cfg['trained_at'],
            'n_features':        cfg['n_features'],
        },
    }


def probability_distribution(bins: int = 20) -> dict:
    """Histogram-ready bin edges + counts across the active book.

    Filtered to active accounts: including already-churned accounts produces a
    huge tall-right-tail spike (their probabilities cluster near 1.0) that
    visually drowns out the actionable distribution.
    """
    active = _active_subset(_load_scoring_frame())
    counts, edges = np.histogram(active[_PROB_COL].values, bins=bins, range=(0.0, 1.0))
    return {
        'bin_edges': [float(e) for e in edges],
        'counts':    [int(c) for c in counts],
        'threshold': float(_load_config()['threshold']),
    }