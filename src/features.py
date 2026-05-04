"""Deterministic feature engineering pipeline.

Preprocesses the three source frames (ALE, UEM, SIH) a standard frame indexed by
`account_id` and consumed by the locked model.

Data-fitted selection (variance / MI / chi^2 / L1 / permutation pruning) is not
performed here, just preprocessing

    pp = build_feature_frame(ale_df, uem_df, sih_df)
    X  = pp.drop(columns=['account_status']).reindex(columns=feature_columns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import (
    ALE_TIMESTAMP_DROPS,
    ALE_TIMEZONE_COLUMN_MAP,
    BINARY_EQUALITY_COLS,
    BOOLEAN_INT_COLS,
    DEFAULT_SNAPSHOT_DATE,
    ID_DROPS_POST_MERGE,
    LEAKAGE_DROPS_POST_MERGE,
    NOTIF_PREF_MAP,
    ONE_HOT_COLS,
    ORDINAL_MAPS,
    PRESERVE_NAN_COLS,
    PRIMARY_ID,
    PRIORITY_MAP,
    SENTIMENT_MAP,
    SIH_TIMEZONE_COLUMN_MAP,
    THRESHOLD_BANDS,
    UEM_TIMEZONE_COLUMN_MAP,
)

# ---------------------------------------------------------------------------
# Per-source preprocessing
# ---------------------------------------------------------------------------

def _timestamps_to_durations(df: pd.DataFrame, column_map: dict[str, str], reference_date: pd.Timestamp) -> None:
    # Add signed integer day-offset columns in-place. Source columns are preserved
    for src_col, dst_col in column_map.items():
        df[dst_col] = (reference_date - df[src_col]).dt.days


def _preprocess_ale(ale: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    ale = ale.copy()
    ale['region'] = ale['region'].fillna('NA') # pandas parses the string 'NA' as null — restore North America region.
    _timestamps_to_durations(ale, ALE_TIMEZONE_COLUMN_MAP, snapshot)
    return ale


def _preprocess_uem(uem: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    uem = uem.copy()
    _timestamps_to_durations(uem, UEM_TIMEZONE_COLUMN_MAP, snapshot)
    return uem


def _preprocess_sih(sih: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    sih = sih.copy()
    _timestamps_to_durations(sih, SIH_TIMEZONE_COLUMN_MAP, snapshot)
    return sih


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def _aggregate_uem(uem: pd.DataFrame) -> pd.DataFrame:
    # Building aggregate fields for the uem frame

    group = uem.groupby(PRIMARY_ID) # Returns a pandas groupby object 

    # TODO missing the `invited` column from data dictionary, could add here if curveball includes it
    # TODO Extract the activity_counters into a helper method for uem and sih
    counts = pd.DataFrame({
        'num_users':             group.size(),
        'num_active_users':      group['user_status'].apply(lambda s: (s == 'active').sum()),
        'num_deactivated_users': group['user_status'].apply(lambda s: (s == 'deactivated').sum()),
    })

    # One-hot-encoding the user_role column
    roles = pd.get_dummies(uem['user_role']).groupby(uem[PRIMARY_ID]).sum()
    roles.columns = [f'num_{c}s' for c in roles.columns]

    # Creating aggregate fields for numerical columns measuring user activity
    activity_counters = {
        'login_count_30d':       ('total_logins_30d',         'avg_logins_30d'),
        'login_count_90d':       ('total_logins_90d',         'avg_logins_90d'),
        'sessions_count_30d':    ('total_sessions_30d',       'avg_sessions_30d'),
        'reports_generated_30d': ('total_reports_30d',        'avg_reports_30d'),
        'dashboards_created':    ('total_dashboards_created', 'avg_dashboards_created'),
        'dashboards_shared':     ('total_dashboards_shared',  'avg_dashboards_shared'),
        'exports_count_30d':     ('total_exports_30d',        'avg_exports_30d'),
    }
    activity_sums = pd.DataFrame({total: group[src].sum()  for src, (total, _)   in activity_counters.items()})
    activity_avgs = pd.DataFrame({avg:   group[src].mean() for src, (_,     avg) in activity_counters.items()})

    fus = group['feature_usage_score'].agg(
        avg_feature_usage_score='mean',
        min_feature_usage_score='min',
        max_feature_usage_score='max',
    )

    def _weighted_session_duration(grp):
        # Weighting average session count by 30d activity (favors recent activity)
        w, v = grp['sessions_count_30d'], grp['avg_session_duration_minutes']
        tot = w.sum()
        return (v * w).sum() / tot if tot > 0 else np.nan
    session_dur = group.apply(_weighted_session_duration).rename('avg_session_duration_minutes')

    # Aggregating TRUEs from boolean fields
    bool_sums = pd.DataFrame({
        'num_api_key_active':        group['api_key_active'].sum(),
        'num_mobile_users':          group['mobile_app_user'].sum(),
        'num_onboarding_completed':  group['onboarding_completed'].sum(),
        'num_certification_earned':  group['certification_earned'].sum(),
        'num_beta_features_enabled': group['beta_features_enabled'].sum(),
    })

    # Calculating spread (variability) in relevant nominal categorical fields
    spread = pd.DataFrame({
        'n_distinct_browsers':  group['browser_type'].nunique(),
        'n_distinct_os':        group['operating_system'].nunique(),
        'n_distinct_languages': group['language_preference'].nunique(),
    })

    # One-hot-encoding nominal categorical groups 
    browser_oh = pd.get_dummies(uem['browser_type'],       prefix='num').groupby(uem[PRIMARY_ID]).sum()
    os_oh      = pd.get_dummies(uem['operating_system'],   prefix='num').groupby(uem[PRIMARY_ID]).sum()
    lang_oh    = pd.get_dummies(uem['language_preference'], prefix='num_lang').groupby(uem[PRIMARY_ID]).sum()

    notif_ord = uem['notification_preference'].map(NOTIF_PREF_MAP)
    notif = pd.DataFrame({
        'total_notification_pref': notif_ord.groupby(uem[PRIMARY_ID]).sum(),
        'avg_notification_pref':   notif_ord.groupby(uem[PRIMARY_ID]).mean(),
    })

    # Creating spread fields for timezone offset
    tz = pd.DataFrame({
        'tz_nunique': group['timezone_offset'].nunique(),
        'tz_std':     group['timezone_offset'].std(),
    })

    profile = group['profile_completeness_pct'].mean().rename('avg_profile_completeness_pct')

    # Ranges for days_since_ fields
    recency = pd.DataFrame({
        'days_since_latest_login':           group['days_since_last_login'].min(),
        'days_since_earliest_login':         group['days_since_last_login'].max(),
        'days_since_latest_user_creation':   group['days_since_user_creation'].min(),
        'days_since_earliest_user_creation': group['days_since_user_creation'].max(),
    })

    # Constructing aggregate columns from UEM frame. Row size should be 5000
    return pd.concat(
        [counts, roles, activity_sums, activity_avgs, fus, session_dur, bool_sums, spread,
         browser_oh, os_oh, lang_oh, notif, tz, profile, recency],
        axis=1,
    )


def _aggregate_sih(sih: pd.DataFrame) -> pd.DataFrame:
    # Building aggregate fields for the sih frame 

    group = sih.groupby(PRIMARY_ID)

    counts = pd.DataFrame({
        'num_tickets':         group.size(),
        'num_ticket_creators': group['user_id'].nunique(),
        'num_distinct_agents': group['agent_id'].nunique(),
    })

    prio_ord = sih['ticket_priority'].map(PRIORITY_MAP)
    sent_ord = sih['ticket_sentiment'].map(SENTIMENT_MAP)
    ordinal = pd.DataFrame({
        'max_priority':  prio_ord.groupby(sih[PRIMARY_ID]).max(),
        'min_priority':  prio_ord.groupby(sih[PRIMARY_ID]).min(),
        'avg_priority':  prio_ord.groupby(sih[PRIMARY_ID]).mean(),
        'max_sentiment': sent_ord.groupby(sih[PRIMARY_ID]).max(),
        'min_sentiment': sent_ord.groupby(sih[PRIMARY_ID]).min(),
        'avg_sentiment': sent_ord.groupby(sih[PRIMARY_ID]).mean(),
    })

    # One-hot-encoding ticket category fields
    cat_oh    = pd.get_dummies(sih['ticket_category']).groupby(sih[PRIMARY_ID])
    cat_sums  = cat_oh.sum().add_prefix('num_')
    cat_rates = cat_oh.mean().add_prefix('rate_')

    resolution = group['resolution_time_hours'].agg(
        max_resolution_hours='max',
        avg_resolution_hours='mean',
        total_resolution_hours='sum',
    )

    satrat = sih['satisfaction_rating']
    satrat_df = pd.DataFrame({
        'num_satrat_responses': satrat.notna().groupby(sih[PRIMARY_ID]).sum(),
        'satrat_response_rate': satrat.notna().groupby(sih[PRIMARY_ID]).mean(),
        'num_low_satrat':       satrat.lt(2.5).groupby(sih[PRIMARY_ID]).sum(),
        'num_high_satrat':      satrat.gt(2.5).groupby(sih[PRIMARY_ID]).sum(),
    })

    bools = pd.DataFrame({
        'total_escalated':               group['escalated'].sum(),
        'rate_escalated':                group['escalated'].mean(),
        'total_sla_breach':              group['sla_breach'].sum(),
        'total_cancellation_requested':  group['cancellation_requested'].sum(),
        'rate_cancellation_requested':   group['cancellation_requested'].mean(),
        'total_retention_offer_made':    group['retention_offer_made'].sum(),
        'total_account_pause_requested': group['account_pause_requested'].sum(),
        'total_downgrade_requested':     group['downgrade_requested'].sum(),
    })

    reopened = group['reopened_count'].agg(total_reopened='sum', max_reopened='max')

    text_counts = pd.DataFrame({
        'num_cancellation_reasons': sih['cancellation_reason'].notna().groupby(sih[PRIMARY_ID]).sum(),
        'num_competitor_mentions':  sih['competitor_mentioned'].notna().groupby(sih[PRIMARY_ID]).sum(),
    })

    # One-hot-encoding the support channel
    ch_oh    = pd.get_dummies(sih['channel']).groupby(sih[PRIMARY_ID])
    ch_sums  = ch_oh.sum().add_prefix('num_')
    ch_rates = ch_oh.mean().add_prefix('rate_')

    activity_counters = {
        'interaction_count':      ('total_interactions', 'avg_interactions'),
        'kb_articles_referenced': ('total_kb_articles',  'avg_kb_articles'),
    }
    activity_sums = pd.DataFrame({total: group[src].sum()  for src, (total, _)   in activity_counters.items()})
    activity_avgs = pd.DataFrame({avg:   group[src].mean() for src, (_,     avg) in activity_counters.items()})

    # internal_category is being phased out per the data dictionary. Serving time data will not match train if used
    if sih['internal_category'].isna().mean() < 0.5:
        ic_oh = pd.get_dummies(sih['internal_category'], prefix='num_internal').groupby(sih[PRIMARY_ID]).sum()
    else:
        ic_oh = pd.DataFrame(index=group.size().index)

    # Handling days_since_ fields
    recency = pd.DataFrame({
        'days_since_latest_ticket_creation':   group['days_since_ticket_creation'].min(),
        'days_since_earliest_ticket_creation': group['days_since_ticket_creation'].max(),
        'days_since_latest_resolution':        group['days_since_resolution'].min(),
        'days_since_earliest_resolution':      group['days_since_resolution'].max(),
    })

    days = sih['days_since_ticket_creation']
    windows = pd.DataFrame({
        'n_tickets_30d': days.le(30).groupby(sih[PRIMARY_ID]).sum(),
        'n_tickets_90d': days.le(90).groupby(sih[PRIMARY_ID]).sum(),
    })

    # Returning aggregated columns from sih (row size should be 5000)
    return pd.concat(
        [counts, ordinal, cat_sums, cat_rates, resolution, satrat_df, bools, reopened,
         text_counts, ch_sums, ch_rates, activity_sums, activity_avgs, ic_oh, recency, windows],
        axis=1,
    )

# ---------------------------------------------------------------------------
# Merge + post-merge cleaning
# ---------------------------------------------------------------------------

def _merge(base: pd.DataFrame, merge_col: str, agg_frames: list[pd.DataFrame]) -> pd.DataFrame:
    # Iteratively left-merge each aggregate frame onto base on `merge_col`.
    # Always join on account_id — account_uuid is NULL for non-EU / pre-2023 rows
    # (data dictionary; CLAUDE.md).
    merged_frame = base
    for frame in agg_frames:
        merged_frame = merged_frame.merge(frame, how='left', on=merge_col)
    return merged_frame


def _zero_fill_after_merge(pp_df: pd.DataFrame, columns) -> None:
    # Replace NaN with 0 in place for `columns`. Use after left-merging aggregate
    # frames where missing-on-the-right means "zero of that thing" (zero tickets,
    # zero sessions, single-user account, ...). Caller is responsible for excluding
    # columns where NaN is semantically meaningful (see PRESERVE_NAN_COLS).
    pp_df[columns] = pp_df[columns].fillna(0)


def _post_merge_clean(pp_df: pd.DataFrame) -> None:
    assert pp_df[PRIMARY_ID].is_unique, "row inflation from merge"

    # Threshold bands run BEFORE drops so a band's source column can be dropped as leakage.
    for output_col, (source_col, threshold) in THRESHOLD_BANDS.items():
        pp_df[output_col] = (pp_df[source_col] >= threshold).astype(int)

    pp_df.drop(
        columns=[*ID_DROPS_POST_MERGE, *LEAKAGE_DROPS_POST_MERGE, *ALE_TIMESTAMP_DROPS],
        inplace=True,
        errors='ignore',
    )

    for col, positive in BINARY_EQUALITY_COLS.items():
        pp_df[col] = (pp_df[col] == positive).astype(int)

    for col, mapping in ORDINAL_MAPS.items():
        pp_df[col] = pp_df[col].map(mapping)

    for col in ONE_HOT_COLS:
        oh = pd.get_dummies(pp_df[col], prefix=col).astype(int)
        pp_df.drop(columns=[col], inplace=True)
        for new_col in oh.columns:
            pp_df[new_col] = oh[new_col].values

    pp_df[BOOLEAN_INT_COLS] = pp_df[BOOLEAN_INT_COLS].astype(int)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_feature_frame(
    ale_df: pd.DataFrame,
    uem_df: pd.DataFrame,
    sih_df: pd.DataFrame,
    snapshot_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Raw three-table input -> one-row-per-account feature frame.

    Returns a preprocessed dataframe with all derived/encoded columns and a binarized target.
    Returned columns are a superset of the locked feature list. Callers should reindex
    based on models/feature_columns.json

    Input frames are not mutated — each is copied once on entry.
    """
    snapshot = pd.Timestamp(snapshot_date) if snapshot_date is not None else DEFAULT_SNAPSHOT_DATE

    ale = _preprocess_ale(ale_df, snapshot)
    uem = _preprocess_uem(uem_df, snapshot)
    sih = _preprocess_sih(sih_df, snapshot)

    agg_frames = [_aggregate_uem(uem), _aggregate_sih(sih)]

    pp_df = _merge(ale, merge_col=PRIMARY_ID, agg_frames=agg_frames)

    agg_cols = [c for f in agg_frames for c in f.columns]
    zero_fill_cols = [c for c in agg_cols if c not in PRESERVE_NAN_COLS]
    _zero_fill_after_merge(pp_df, zero_fill_cols)

    _post_merge_clean(pp_df)
    return pp_df
