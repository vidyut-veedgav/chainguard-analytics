"""Model training, threshold tuning, and artifact persistence.

    - A swappable model factory (defaults to the XGBoost config proved to work in EDA)
    - Stratified split + F-beta threshold sweep on a held-out fold
    - Refit on the full labeled dataset before saving (matches notebook cell 57)
    - Persistence of model weights + a config.json (same format as current)

Phase-2 levers (all in `src/config.py`):
    - Swap model + hyperparameters: pass `model_factory=` (or edit
      `XGB_HYPERPARAMS`). Everything else (split, sweep, save) stays put.
    - Shift the precision/recall trade-off: change `DEFAULT_BETA` (F2 = recall
      weighted 4x precision, matches the notebook).
    - Save under a different name to compare runs: pass `model_path=` /
      `config_path=` to `save_artifacts`, or change `MODEL_WEIGHTS_PATH` /
      `MODEL_CONFIG_PATH`.
    - Non-XGBoost estimator: `save_artifacts` assumes `estimator.save_model(path)`.
      For sklearn classifiers, the one-line override is `joblib.dump(estimator, path)`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import (
    DEFAULT_BETA,
    DEFAULT_THRESHOLDS,
    MODEL_CONFIG_PATH,
    MODEL_WEIGHTS_PATH,
    TRAINING_RANDOM_STATE,
    TRAINING_TEST_SIZE,
    XGB_HYPERPARAMS,
)
from src._validation import require_in_range, require_nonempty


def default_model_factory() -> XGBClassifier:
    # XGB_HYPERPARAMS is the EDA-locked config
    return XGBClassifier(**XGB_HYPERPARAMS)


# ---------------------------------------------------------------------------
# Threshold sweep helpers
# ---------------------------------------------------------------------------

def _fbeta(precision: float, recall: float, beta: float) -> float:
    # Comptutes F-beta score, measuring combined precision and recall with tunable beta to controll tradeoff
    # Used because accuracy can be misleading with imbalanced data
    denom = (beta * beta * precision) + recall
    return (1 + beta * beta) * precision * recall / denom if denom else 0.0


def _sweep_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray,
    beta: float,
) -> pd.DataFrame:
    # Model defaults to 0.5 as the classification labelling threshold but since we also care about 
    # maximizing recall, we want to be able to tune this. For the saved XGB model, the best threshold was 0.10
    # It's better to flag accounts that are less likely to churn than miss accounts that will churn
    # F-beta score lets us bias toward recall without going to the extreme
    fbeta_key = f'f{beta:g}'
    rows = []
    for t in thresholds:
        y_hat = (y_proba >= t).astype(int)
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append({
            'threshold': round(float(t), 2),
            'flagged':   int((y_hat == 1).sum()),
            'TP': tp, 'FP': fp, 'FN': fn,
            'precision': round(prec, 4),
            'recall':    round(rec, 4),
            'f1':        round(_fbeta(prec, rec, 1.0), 4),
            fbeta_key:   round(_fbeta(prec, rec, beta), 4),
        })
    return pd.DataFrame(rows)


def _best_threshold(sweep: pd.DataFrame, beta: float) -> float:
    # Chooses 
    return float(sweep.loc[sweep[f'f{beta:g}'].idxmax(), 'threshold'])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    model_factory: Callable[[], object] = default_model_factory,
    test_size: float = TRAINING_TEST_SIZE,
    random_state: int = TRAINING_RANDOM_STATE,
) -> dict:
    """Stratified split + fit on the train fold.

    Returns:
        {
          'estimator':    <fitted on train fold>,
          'y_test':       np.ndarray,     # held-out labels
          'proba':        np.ndarray,     # class-1 probabilities on the held-out fold
          'test_size':    float,          # echoed for tune_threshold's note
          'random_state': int,            # echoed for tune_threshold's note
        }
    """
    require_nonempty(X, where="train")
    require_in_range("test_size", test_size, 0.0, 1.0, where="train")
    y_arr = np.asarray(y).ravel()
    if len(X) != len(y_arr):
        raise ValueError(f"train: X has {len(X)} rows, y has {len(y_arr)} — must match")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_arr, test_size=test_size, stratify=y_arr, random_state=random_state,
    )
    estimator = model_factory()
    estimator.fit(X_tr, y_tr)
    proba = estimator.predict_proba(X_te)[:, 1]
    return {
        'estimator':    estimator,
        'y_test':       y_te,
        'proba':        proba,
        'test_size':    test_size,
        'random_state': random_state,
    }


def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    beta: float = DEFAULT_BETA,
    thresholds: np.ndarray = DEFAULT_THRESHOLDS,
    test_size: float | None = None,
    random_state: int | None = None,
) -> dict:
    """F-beta sweep over thresholds; pick the arg-max.

    Returns:
        {
          'threshold':       float,
          'sweep':           pd.DataFrame,      # threshold, TP/FP/FN, precision, recall, f1, f<beta>
          'holdout_metrics': dict,              # shape matches models/config.json["holdout_metrics"]
        }
    """
    if beta <= 0:
        raise ValueError(f"tune_threshold: beta must be > 0, got {beta}")
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    if len(y_true) != len(y_proba):
        raise ValueError(f"tune_threshold: y_true ({len(y_true)}) and y_proba ({len(y_proba)}) must match")
    sweep = _sweep_thresholds(y_true, y_proba, thresholds, beta)
    threshold = _best_threshold(sweep, beta)
    y_hat = (y_proba >= threshold).astype(int)

    fbeta_key = f'f{beta:g}'
    holdout_metrics: dict = {}
    if test_size is not None and random_state is not None:
        holdout_metrics['note'] = (
            f'computed on a single {int((1 - test_size) * 100)}/{int(test_size * 100)} '
            f'stratified split (random_state={random_state}); reported for reference. '
            f'Final model above is refit on full data.'
        )
    holdout_metrics.update({
        'n_test_rows':       int(len(y_true)),
        'n_test_positives':  int(y_true.sum()),
        'average_precision': round(float(average_precision_score(y_true, y_proba)), 4),
        'at_threshold': {
            'threshold': float(threshold),
            'precision': round(float(precision_score(y_true, y_hat)), 4),
            'recall':    round(float(recall_score(y_true, y_hat)), 4),
            'f1':        round(float(f1_score(y_true, y_hat)), 4),
            fbeta_key:   round(float(fbeta_score(y_true, y_hat, beta=beta)), 4),
        },
    })
    return {
        'threshold':       float(threshold),
        'sweep':           sweep,
        'holdout_metrics': holdout_metrics,
    }


def fit_final(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    model_factory: Callable[[], object] = default_model_factory,
):
    """Refit a new estimator on the full labeled dataset.

    Threshold is already locked from `train_and_tune`; this just spends the
    held-out rows to maximize information used by the deployed model.
    """
    estimator = model_factory()
    estimator.fit(X, np.asarray(y).ravel())
    return estimator


def save_artifacts(
    estimator,
    *,
    threshold: float,
    holdout_metrics: dict,
    n_features: int,
    n_training_rows: int,
    beta: float = DEFAULT_BETA,
    model_name: str = 'xgboost',
    model_path: Path | str = MODEL_WEIGHTS_PATH,
    config_path: Path | str = MODEL_CONFIG_PATH,
) -> None:
    """Persist model weights + config.json (schema matches existing models/config.json).

    Note: assumed an XGBoost style json, could be different for an sklearn model
    """
    model_path  = Path(model_path)
    config_path = Path(config_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    estimator.save_model(model_path)

    config = {
        'model':           model_name,
        'trained_at':      datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'n_training_rows': int(n_training_rows),
        'n_features':      int(n_features),
        'threshold':       float(threshold),
        'beta':            float(beta),
        'holdout_metrics': holdout_metrics,
    }
    with config_path.open('w') as f:
        json.dump(config, f, indent=2)
