"""Module-level constants for the deterministic feature pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Join key across all source frames + per-account aggregation key.
PRIMARY_ID = 'account_id'

# Snapshot date threshold computed from existing timestamps across the source frames.
DEFAULT_SNAPSHOT_DATE = pd.Timestamp("2025-01-02")

# Outcome-encoding fields flagged by the data dictionary.
# Source columns referenced by THRESHOLD_BANDS are listed here intentionally
# the band is derived before drops, so the source can also be dropped here as leakage.
LEAKAGE_DROPS_POST_MERGE = ['days_since_status_change', 'account_health_score']

# No need for IDs in feature set.
ID_DROPS_POST_MERGE = [PRIMARY_ID, 'org_id', 'account_manager_id',
                       'account_uuid', 'data_residency_region']

# Timestamps are converted to days_since_ columns; the originals can be safely dropped.
ALE_TIMESTAMP_DROPS = ['created_timestamp', 'last_activity_timestamp',
                       'status_change_date', 'contract_end_date']

# Mapping timestamp columns to their derived days_since_ counterparts.
ALE_TIMEZONE_COLUMN_MAP = {
    'created_timestamp':       'days_since_creation',
    'last_activity_timestamp': 'days_since_last_activity',
    'contract_end_date':       'days_since_contract_end',
    'status_change_date':      'days_since_status_change',
}
UEM_TIMEZONE_COLUMN_MAP = {
    'last_login_date':   'days_since_last_login',
    'user_created_date': 'days_since_user_creation',
}
SIH_TIMEZONE_COLUMN_MAP = {
    'created_date':  'days_since_ticket_creation',
    'resolved_date': 'days_since_resolution',
}

SUBSCRIPTION_TIER_ORDINAL = {
    'free': 1,
    'starter': 2,
    'professional': 3,
    'enterprise': 4,
}
PRIORITY_MAP = {
    'low': 0,
    'medium': 1,
    'high': 2,
    'critical': 3,
}
SENTIMENT_MAP = {
    'frustrated': 0,
    'negative': 1,
    'neutral': 2,
    'positive': 3,
}
NOTIF_PREF_MAP = {
    'none': 0,
    'important': 1,
    'all': 2,
}
BOOLEAN_INT_COLS = [
    'auto_renew_enabled',
    'risk_flag',
    'api_calls_enabled',
    'sso_enabled',
    'white_label_enabled',
]
ONE_HOT_COLS = [
    'region',
    'billing_cycle',
    'payment_method',
    'provisioning_method',
]
TARGET = 'account_status'

# After a left-merge, missing-on-the-right rows get NaN (e.g. account_id in sih). For most aggregate columns,
# that means there are zero of that thing and the NaN should become 0. These columns exceptions (NaN is semantically meaningful)
PRESERVE_NAN_COLS = [
    'days_since_latest_ticket_creation',
    'days_since_earliest_ticket_creation',
    'days_since_latest_resolution',
    'days_since_earliest_resolution',
]

# Threshold-band derivations: output_col -> (source_col, threshold).
# Each output is `(source >= threshold).astype(int)` and is derived before the post-merge drops
# This matters in cases where the banded column is also leaky (e.g. health_score_band)
THRESHOLD_BANDS = {
    # 50 is the midpoint of the 40-60 dead-band the Q3-2023 algorithm-change introduced.
    'health_score_band': ('account_health_score', 50),
}

# Binary-equality encodings: column -> positive value. Use for the prediction target and for any other
# multi-value categorical that should collapse to a single 0/1 indicator.
BINARY_EQUALITY_COLS = {
    TARGET: 'churned',
}

# Ordinal-encoding map: column -> {value: rank}. The column is overwritten with the
# mapped integers. Values not in the mapping become NaN, which surfaces as a loud
# failure if a Phase-2 source ships a new level (e.g. a 'platinum' tier).
ORDINAL_MAPS = {
    'subscription_tier': SUBSCRIPTION_TIER_ORDINAL,
}

# ---------------------------------------------------------------------------
# Feature selection thresholds (consumed by src/features.py)
# ---------------------------------------------------------------------------

# Variance prune. Binary cols where one class exceeds this share are near-constant.
BINARY_DOMINANCE_THRESHOLD = 0.99
# Continuous cols below either floor are near-constant. CV is scale-aware,
# std catches degenerate-zero-variance columns where the mean is also ~0.
CV_THRESHOLD = 1e-3
STD_THRESHOLD = 1e-9

# Pairwise correlation prune. Spearman is intentional — catches monotonic
# redundancy (totals vs averages) without assuming linearity.
CORRELATION_THRESHOLD = 0.9

# Univariate hard-drop. A feature must fail BOTH the nonlinear test (MI < floor)
# AND its applicable linear/independence test (chi² for binary, ANOVA F for
# continuous) at this quantile before it is dropped. Conservative by design.
MI_FLOOR = 1e-3
UNIVARIATE_QUANTILE = 0.25

# Model-based lock-in. Permutation importance below this floor contributes
# < 0.1% AUC and is treated as noise.
PERM_FLOOR = 1e-3

# Train/test split for the model-based stage.
TEST_SIZE = 0.25
RANDOM_STATE = 0

# rate_* siblings carry the same signal more cleanly than these num_* counts;
# excluded at lock-in regardless of L1 / permutation scores to avoid
# double-counting collinear constructs.
NUM_TICKET_DROPS = ['num_billing', 'num_bug', 'num_technical', 'num_feature_request']

# Persisted artifact written by the selection pipeline and read back at serve time.
FEATURE_COLUMNS_PATH = Path('models/feature_columns.json')

# Curated leakage drops applied AFTER deterministic preprocessing but BEFORE
# selection. These are EDA decisions (distribution checks + leakage audit) — not
# fitted, but they require the post-merge frame to identify. Without dropping
# these, the model-based stage trivially achieves AUC ≈ 1.0 and the lock-in
# collapses to a single perfect-separator (e.g., health_score_band).
#
# - health_score_band: encodes account_health_score (algorithm proxies churn risk)
# - risk_flag: CS-team churn assessment, NULL for ~50% of accounts
# - days_since_last_activity: post-churn drift (median 63 churned vs 18 active)
# - num_competitor_mentions: exit-conversation flag (100% zero active vs 78% nonzero churned)
# - total_/rate_cancellation_*, num_cancellation*, total_retention_offer_made,
#   total_account_pause_requested, total_downgrade_requested: SIH CS-conversation
#   artifacts that telegraph the outcome.
POST_PREPROCESSING_LEAKAGE_DROPS = [
    'health_score_band',
    'risk_flag',
    'days_since_last_activity',
    'num_competitor_mentions',
    'total_cancellation_requested',
    'rate_cancellation_requested',
    'num_cancellation',
    'rate_cancellation',
    'num_cancellation_reasons',
    'total_retention_offer_made',
    'total_account_pause_requested',
    'total_downgrade_requested',
]
