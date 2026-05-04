"""Feature selection pipeline.

Implements a four-stage feature selection pipeline

    1. variance prune          (binary dominance, continuous CV / std)
    2. pairwise correlation    (spearman, dropping the weaker-target side)
    3. univariate filter       (MI floor AND chi², ANOVA F at bottom quartile)
    4. model-based selection   (L1 logistic regression + RF permutation importance)
    5. lock-in                 (in_both AND perm >= floor, minus NUM_TICKET_DROPS)

    # TODO learn how MI, chi^2 and ANOVA F work
    # TODO learn how RF permutation importance works
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    BINARY_DOMINANCE_THRESHOLD,
    CORRELATION_THRESHOLD,
    CV_THRESHOLD,
    FEATURE_COLUMNS_PATH,
    MI_FLOOR,
    NUM_TICKET_DROPS,
    PERM_FLOOR,
    POST_PREPROCESSING_LEAKAGE_DROPS,
    RANDOM_STATE,
    STD_THRESHOLD,
    TARGET,
    TEST_SIZE,
    UNIVARIATE_QUANTILE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_binary(s: pd.Series) -> bool:
    # dtype-based detection misses int-encoded booleans and one-hots.
    return set(s.dropna().unique()).issubset({0, 1})


def _split_binary_continuous(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    # Classify features as either binary or continuous
    binary_cols, continuous_cols = [], []
    for c in features.columns:
        (binary_cols if _is_binary(features[c]) else continuous_cols).append(c)
    return binary_cols, continuous_cols


def _median_impute(features: pd.DataFrame) -> pd.DataFrame:
    # For diagnostics since some features carry NaNs as in-built signal (e.g. recency for accounts with no tickets)
    return features.fillna(features.median(numeric_only=True))


# ---------------------------------------------------------------------------
# Stage 1 — Variance prune
# ---------------------------------------------------------------------------

def _filter_variance(features: pd.DataFrame) -> list[str]:
    # Filter columns based on variance. Super low variance gives low predictive signal
    # Using cv = sd / mu because it's scale aware. mu 1000, st 0.001 is very stable but mu 0.001 and st 0.001 is not
    # Out of the box sklearn would flatten to 1 and kill thresholding
    binary_cols, continuous_cols = _split_binary_continuous(features)

    binary_share = features[binary_cols].apply(lambda s: max(s.mean(), 1 - s.mean()))
    binary_drops = binary_share[binary_share > BINARY_DOMINANCE_THRESHOLD].index.tolist()

    mu = features[continuous_cols].mean().abs()
    sd = features[continuous_cols].std()
    cv = sd / mu.replace(0, np.nan)
    cont_drops = cv[(cv < CV_THRESHOLD) | (sd < STD_THRESHOLD)].index.tolist()

    return binary_drops + cont_drops


# ---------------------------------------------------------------------------
# Stage 2 — Pairwise correlation prune
# ---------------------------------------------------------------------------

def _prune_correlation(
    # Measure the correlation between different features to find redundancies
    # Using spearman correlation over pearson because it handles monotonic relationships instead of just linear 
    # and is more robust with outliers since it uses data ranks instead of raw data
    features: pd.DataFrame,
    y: pd.Series,
    method: str = 'spearman',
    threshold: float = CORRELATION_THRESHOLD,
) -> list[str]:
    X_imp = _median_impute(features)
    corr = X_imp.corr(method=method).abs()
    target_corr = X_imp.corrwith(y, method=method).abs()

    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop: set[str] = set()
    for col in upper.columns:
        for row, val in upper[col].dropna().items():
            if val <= threshold or row in to_drop or col in to_drop:
                continue
            # Drop the side with weaker target correlation; the other survives.
            loser = row if target_corr[row] < target_corr[col] else col
            to_drop.add(loser)
    return sorted(to_drop)


# ---------------------------------------------------------------------------
# Stage 3 — Univariate filters (ANOVA F, chi², mutual information)
# ---------------------------------------------------------------------------

def _rank_univariate(
    features: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    # Evaluates each feature independently against the target and returns three importance scores
    # ANOVA F: tests how different the average values of each class in a feature are (variance between variables is greated than variance within variable --> reject null)
    # Chi-square: tests whether a binary feature and the target are independent
    # Mutual information: noisier but a helpful filter for extremes, measures linear and nonlinear dependencies / interactions
    X_imp = _median_impute(features)
    binary_cols, continuous_cols = _split_binary_continuous(X_imp)

    # ANOVA F on continuous (chi² requires non-negative inputs; some continuous cols can be negative)
    f_stat, _ = f_classif(X_imp[continuous_cols], y)
    anova = pd.Series(f_stat, index=continuous_cols, name='anova_F')

    # chi² on binary one-hots (already non-negative by construction)
    chi_stat, _ = chi2(X_imp[binary_cols], y)
    chi = pd.Series(chi_stat, index=binary_cols, name='chi2')

    # Mutual information across everything; flag binaries as discrete for kNN-density estimator
    mi = pd.Series(
        mutual_info_classif(
            X_imp, y,
            discrete_features=[c in binary_cols for c in X_imp.columns],
            random_state=RANDOM_STATE,
        ),
        index=X_imp.columns, name='mutual_info',
    )
    return anova, chi, mi


def _univariate_drops(
    anova: pd.Series,
    chi: pd.Series,
    mi: pd.Series,
    mi_floor: float = MI_FLOOR,
    quantile: float = UNIVARIATE_QUANTILE,
) -> list[str]:
    # Hard drop only when MI agrees with the applicable secondary test since it's noisier
    # Features need to fail both a nonlinear and a linear/independence test to be dropped
    anova_q = anova.quantile(quantile)
    chi_q = chi.quantile(quantile)
    drops = []
    for c in mi.index:
        if mi[c] >= mi_floor:
            continue
        if c in chi.index and chi[c] < chi_q:
            drops.append(c)
        elif c in anova.index and anova[c] < anova_q:
            drops.append(c)
    return drops


def _build_rank_table(anova: pd.Series, chi: pd.Series, mi: pd.Series) -> pd.DataFrame:
    # Construct a ranking of features that independently correlate with target
    cols = mi.index
    table = pd.DataFrame(index=cols)
    table['anova_F_rank'] = anova.reindex(cols).rank(ascending=False)
    table['chi2_rank']    = chi.reindex(cols).rank(ascending=False)
    table['mi_rank']      = mi.rank(ascending=False)
    table['mi']           = mi
    table['avg_rank']     = table[['anova_F_rank', 'chi2_rank', 'mi_rank']].mean(axis=1)
    return table.sort_values('avg_rank')


# ---------------------------------------------------------------------------
# Stage 4 — Model-based selection (two estimators, two views)
# ---------------------------------------------------------------------------

def _fit_l1_selector(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
) -> tuple[pd.Series, float]:
    # Trains a logistic regression model with L1 regularization to assess feature importance
    # Use L1 reg. because it sends feature weights get sent to 0 instead of tiny floats, acting as a built-in feature selector
    # L1 needs scale-comparable inputs so fitting scaler on train set only is important to aviod leakage
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    l1 = LogisticRegressionCV(
        Cs=10, penalty='l1', solver='saga', scoring='roc_auc',
        cv=5, max_iter=5000, class_weight='balanced',
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    l1.fit(X_tr_s, y_tr)
    coef = pd.Series(l1.coef_.ravel(), index=X_tr.columns, name='l1_coef')
    auc = roc_auc_score(y_te, l1.predict_proba(X_te_s)[:, 1])
    return coef, auc


def _fit_rf_permutation(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
) -> tuple[pd.Series, float]:
    # Trains a random forest classifier then measuring feature importance using permutation importance
    # Perm. imporance takes a feature, shuffles its values, and measure how much model performance drops (bigger drops = more important features)
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=20,
        class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
    perm = permutation_importance(
        rf, X_te, y_te,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1, scoring='roc_auc',
    )
    importance = pd.Series(perm.importances_mean, index=X_tr.columns, name='perm_importance')
    return importance, auc


def _combine_scores(l1_coef: pd.Series, perm_importance: pd.Series) -> pd.DataFrame:
    # Combining results from both models
    combined = pd.DataFrame({
        'l1_coef':         l1_coef,
        'l1_abs':          l1_coef.abs(),
        'perm_importance': perm_importance,
    })
    combined['in_l1']   = combined['l1_abs'] > 1e-6
    combined['in_perm'] = combined['perm_importance'] > 0
    combined['in_both'] = combined['in_l1'] & combined['in_perm']
    return combined.sort_values(['in_both', 'perm_importance', 'l1_abs'], ascending=False)


# ---------------------------------------------------------------------------
# Stage 5 — Lock-in
# ---------------------------------------------------------------------------

def _lock_features(
    combined: pd.DataFrame,
    perm_floor: float = PERM_FLOOR,
    exclusions: Iterable[str] | None = None,
) -> list[str]:
    # Create a final filtered list of features
    exclusions = set(exclusions if exclusions is not None else NUM_TICKET_DROPS)
    keepers = combined[combined['in_both'] & (combined['perm_importance'] >= perm_floor)].index.tolist()
    keepers = [c for c in keepers if c not in exclusions]
    return sorted(keepers)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _write_feature_columns(features: list[str], path: Path, force: bool = False) -> None:
    # Persists final feature list to specified path, refusing for divergent columns for protection
    if path.exists() and not force:
        existing = _read_feature_columns(path)
        if existing != features:
            only_new = sorted(set(features) - set(existing))
            only_old = sorted(set(existing) - set(features))
            raise ValueError(
                f"refusing to overwrite {path} with a divergent feature list. "
                f"only in new: {only_new or '<none>'}; only in old: {only_old or '<none>'}. "
                f"Pass force=True to confirm re-derivation, or persist_path=None to skip writing."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as fh:
        json.dump(features, fh, indent=2)


def _read_feature_columns(path: Path) -> list[str]:
    with path.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_selection_pipeline(
    pp_df: pd.DataFrame,
    target: str = TARGET,
    persist_path: Path | str | None = FEATURE_COLUMNS_PATH,
    random_state: int = RANDOM_STATE,
    force: bool = False,
) -> dict:
    """Run the four-stage selection funnel against a candidate frame.

    Operates on a working copy; does not mutate input. Splits stratified on y
    for the model-based stage so train/test class balance matches the full data.

    Persistence is conservative: when `persist_path` points to an existing file
    whose contents differ from the new locked list, the call raises ValueError
    unless `force=True`. This prevents a silent rewrite from breaking inference
    on a model that was trained against the prior locked list. Pass `force=True`
    for an intentional re-derivation or `persist_path=None` to compute without writing.

    Returns a diagnostic dict:

        {
          'stage_drops':     {'leakage': [...], 'variance': [...], 'correlation': [...], 'univariate': [...]},
          'rank_table':      DataFrame,        # post-univariate ranking (anova/chi²/mi)
          'scores':          DataFrame,        # l1_coef, perm_importance, in_l1, in_perm, in_both
          'auc':             {'l1': float, 'rf': float},
          'locked_features': [...],
        }
    """
    work = pp_df.copy()
    features = work.drop(columns=[target])
    y = work[target]

    # Known leakage drops applied before any scoring so they cannot dominate the selection stages.
    leakage_drops = [c for c in POST_PREPROCESSING_LEAKAGE_DROPS if c in features.columns]
    features = features.drop(columns=leakage_drops)

    variance_drops = _filter_variance(features)
    features = features.drop(columns=variance_drops)

    correlation_drops = _prune_correlation(features, y)
    features = features.drop(columns=correlation_drops)

    anova, chi, mi = _rank_univariate(features, y)
    univariate_drops = _univariate_drops(anova, chi, mi)
    features = features.drop(columns=univariate_drops)

    rank_table = _build_rank_table(
        anova.drop(univariate_drops, errors='ignore'),
        chi.drop(univariate_drops, errors='ignore'),
        mi.drop(univariate_drops, errors='ignore'),
    )

    X_imp = _median_impute(features)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_imp, y, test_size=TEST_SIZE, stratify=y, random_state=random_state,
    )
    l1_coef, l1_auc = _fit_l1_selector(X_tr, y_tr, X_te, y_te)
    perm_importance, rf_auc = _fit_rf_permutation(X_tr, y_tr, X_te, y_te)
    scores = _combine_scores(l1_coef, perm_importance)

    locked = _lock_features(scores)

    if persist_path is not None:
        _write_feature_columns(locked, Path(persist_path), force=force)

    return {
        'stage_drops': {
            'leakage': leakage_drops,
            'variance': variance_drops,
            'correlation': correlation_drops,
            'univariate': univariate_drops,
        },
        'rank_table': rank_table,
        'scores': scores,
        'auc': {'l1': l1_auc, 'rf': rf_auc},
        'locked_features': locked,
    }


def apply_locked_features(
    pp_df: pd.DataFrame,
    feature_columns: list[str] | str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Reindex a preprocessed frame onto the locked feature set.

    `feature_columns` can be a list (used directly), a path (loaded as JSON),
    or None (loads from FEATURE_COLUMNS_PATH). reindex() is intentional: any
    column missing from `pp_df` becomes a NaN column and trips the assert
    """
    if feature_columns is None:
        cols = _read_feature_columns(FEATURE_COLUMNS_PATH)
    elif isinstance(feature_columns, (str, Path)):
        cols = _read_feature_columns(Path(feature_columns))
    else:
        cols = list(feature_columns)

    X = pp_df.reindex(columns=cols)
    missing = X.columns[X.isna().any()].tolist()
    assert not missing, f"locked features missing or NaN at serve time: {missing}"
    y = pp_df[TARGET]
    return X, y
