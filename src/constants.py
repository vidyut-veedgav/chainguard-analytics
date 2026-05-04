"""Module-level constants for the deterministic feature pipeline.
"""

from __future__ import annotations

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
