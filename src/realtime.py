"""Real-time event ingestion onto the preprocessed feature frame.

Consumes batches from the ChurnGuard Events API (see docs/churnguard_api.md) and
folds them into a fresh copy of pp_df. Designed to run BEFORE feature selection
so new realtime signals can earn their way into the locked feature list on the
next retrain.

Two column families are touched:

  - Existing pp_df columns (Class A: absolute state). integration_count is
    incremented/decremented. seats_active/seats_purchased are set from the
    event's `total_seats` metadata. These have no window/temporal-anchor
    conflict; the count IS the current state.

  - New rt_* columns (Class B/C: activity + cumulative-with-realtime-anchor).
    All other event types write here. The snapshot-era windowed columns
    (total_logins_30d, n_tickets_30d, days_since_latest_login, ...) are
    intentionally NOT touched — their window is anchored to
    DEFAULT_SNAPSHOT_DATE, and overwriting them with realtime-era values would
    silently corrupt the model's learned distributions.

Public entry point:

    pp_df_new = apply_events(pp_df, events)

Always returns a new frame; never mutates the input. Events are deduped by
event_id and sorted by timestamp before any strategy runs. Events whose
account_id is not in pp_df.index are skipped with a warning.

For distinct-count features (rt_distinct_users_logged_in,
rt_distinct_features_used) a module-level _seen dict tracks the set per
account. This is in-memory only; on process restart the API replays the full
event history (per the spec) so the counts self-recover.
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from src.config import EVENT_MAPPING, PRIORITY_MAP, SENTIMENT_MAP

logger = logging.getLogger(__name__)

# account_id -> bucket_key -> set of values seen. Rebuilds on process restart
# via API replay (see module docstring).
_seen: dict[str, dict[str, set]] = {}

_TICKET_CATEGORIES = ('billing', 'bug', 'technical', 'feature_request')


# ---------------------------------------------------------------------------
# Dedup + sort
# ---------------------------------------------------------------------------

def _dedup_and_sort(events: list[dict]) -> list[dict]:
    # event_id is the dedup key per the API spec. Sort by timestamp then
    # event_id for stable ordering when timestamps tie.
    by_id = {e['event_id']: e for e in events}
    return sorted(by_id.values(), key=lambda e: (e['timestamp'], e['event_id']))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_event(event: dict) -> None:
    for field in ('event_id', 'event_type', 'account_id', 'timestamp'):
        if field not in event:
            raise ValueError(f"event missing required field {field!r}: {event}")
    if event['event_type'] not in EVENT_MAPPING:
        raise ValueError(f"unknown event_type {event['event_type']!r}")


def _validate_postcondition(
    before: pd.Series,
    after: pd.Series,
    allowed: list[str],
    event_type: str,
) -> None:
    # Confirm the strategy only wrote to columns it declared in EVENT_MAPPING.
    allowed_set = set(allowed)
    for col in after.index:
        b, a = before[col], after[col]
        if pd.isna(b) and pd.isna(a):
            continue
        if b == a:
            continue
        if col not in allowed_set:
            raise AssertionError(
                f"strategy for {event_type!r} mutated undeclared column {col!r}"
            )


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def _track_distinct(account_id: str, key: str, value) -> int:
    bucket = _seen.setdefault(account_id, {}).setdefault(key, set())
    bucket.add(value)
    return len(bucket)


def _update_recency(pp_df: pd.DataFrame, aid: str, col: str, days: int, is_first: bool) -> None:
    # rt_days_since_latest_* columns init to 0. min(0, x) = 0 for x >= 0,
    # so the first event has to overwrite rather than take the min.
    pp_df.at[aid, col] = days if is_first else min(int(pp_df.at[aid, col]), days)


def _update_ordinal_stats(
    pp_df: pd.DataFrame, aid: str,
    min_col: str, max_col: str, avg_col: str,
    value: float, old_n: int, new_n: int,
) -> None:
    # Min/max/avg update for ordinal columns (priority, sentiment). Same
    # first-event sentinel as recency — min initialized to 0 is broken without it.
    if old_n == 0:
        pp_df.at[aid, min_col] = value
        pp_df.at[aid, max_col] = value
        pp_df.at[aid, avg_col] = float(value)
        return
    pp_df.at[aid, min_col] = min(float(pp_df.at[aid, min_col]), value)
    pp_df.at[aid, max_col] = max(float(pp_df.at[aid, max_col]), value)
    old_avg = float(pp_df.at[aid, avg_col])
    pp_df.at[aid, avg_col] = (old_avg * old_n + value) / new_n


# ---------------------------------------------------------------------------
# Strategies — one per event_type
# ---------------------------------------------------------------------------

def _strategy_login(pp_df: pd.DataFrame, event: dict, now: pd.Timestamp) -> None:
    aid = event['account_id']
    is_first = int(pp_df.at[aid, 'rt_total_logins']) == 0
    days = (now - pd.Timestamp(event['timestamp'])).days
    _update_recency(pp_df, aid, 'rt_days_since_latest_login', days, is_first)
    pp_df.at[aid, 'rt_total_logins'] = int(pp_df.at[aid, 'rt_total_logins']) + 1
    user_id = event.get('user_id')
    if user_id:
        pp_df.at[aid, 'rt_distinct_users_logged_in'] = _track_distinct(aid, 'logins_users', user_id)


def _strategy_logout(pp_df: pd.DataFrame, event: dict, now: pd.Timestamp) -> None:
    aid = event['account_id']
    pp_df.at[aid, 'rt_total_logouts'] = int(pp_df.at[aid, 'rt_total_logouts']) + 1
    dur = event.get('metadata', {}).get('session_duration_minutes', 0.0)
    pp_df.at[aid, 'rt_total_session_minutes'] = float(pp_df.at[aid, 'rt_total_session_minutes']) + float(dur)


def _strategy_feature_usage(pp_df: pd.DataFrame, event: dict, now: pd.Timestamp) -> None:
    aid = event['account_id']
    meta = event.get('metadata', {})
    count = int(meta.get('usage_count', 1))
    pp_df.at[aid, 'rt_total_feature_uses'] = int(pp_df.at[aid, 'rt_total_feature_uses']) + count
    feature = meta.get('feature')
    if feature:
        pp_df.at[aid, 'rt_distinct_features_used'] = _track_distinct(aid, 'features', feature)


def _strategy_report_generated(pp_df: pd.DataFrame, event: dict, now: pd.Timestamp) -> None:
    aid = event['account_id']
    rows = int(event.get('metadata', {}).get('rows', 0))
    pp_df.at[aid, 'rt_total_reports'] = int(pp_df.at[aid, 'rt_total_reports']) + 1
    pp_df.at[aid, 'rt_total_report_rows'] = int(pp_df.at[aid, 'rt_total_report_rows']) + rows


def _strategy_dashboard_created(pp_df: pd.DataFrame, event: dict, now: pd.Timestamp) -> None:
    aid = event['account_id']
    widgets = int(event.get('metadata', {}).get('widget_count', 0))
    pp_df.at[aid, 'rt_total_dashboards_created'] = int(pp_df.at[aid, 'rt_total_dashboards_created']) + 1
    pp_df.at[aid, 'rt_total_widgets_created'] = int(pp_df.at[aid, 'rt_total_widgets_created']) + widgets


def _strategy_support_ticket_created(pp_df: pd.DataFrame, event: dict, now: pd.Timestamp) -> None:
    aid = event['account_id']
    meta = event.get('metadata', {})
    old_n = int(pp_df.at[aid, 'rt_num_tickets'])
    new_n = old_n + 1
    pp_df.at[aid, 'rt_num_tickets'] = new_n

    category = meta.get('category')
    if category in _TICKET_CATEGORIES:
        pp_df.at[aid, f'rt_num_{category}'] = int(pp_df.at[aid, f'rt_num_{category}']) + 1
    # Every rate recomputes against the new denominator. Categories outside
    # _TICKET_CATEGORIES leave numerators untouched; their rate still divides
    # correctly by the new total.
    for cat in _TICKET_CATEGORIES:
        pp_df.at[aid, f'rt_rate_{cat}'] = int(pp_df.at[aid, f'rt_num_{cat}']) / new_n

    prio = PRIORITY_MAP.get(meta.get('priority'))
    if prio is not None:
        _update_ordinal_stats(
            pp_df, aid,
            'rt_min_priority', 'rt_max_priority', 'rt_avg_priority',
            prio, old_n, new_n,
        )
    sent = SENTIMENT_MAP.get(meta.get('sentiment'))
    if sent is not None:
        _update_ordinal_stats(
            pp_df, aid,
            'rt_min_sentiment', 'rt_max_sentiment', 'rt_avg_sentiment',
            sent, old_n, new_n,
        )

    days = (now - pd.Timestamp(event['timestamp'])).days
    _update_recency(pp_df, aid, 'rt_days_since_latest_ticket_creation', days, is_first=(old_n == 0))


def _strategy_support_ticket_resolved(pp_df: pd.DataFrame, event: dict, now: pd.Timestamp) -> None:
    aid = event['account_id']
    meta = event.get('metadata', {})
    old_n = int(pp_df.at[aid, 'rt_num_resolved'])
    new_n = old_n + 1
    pp_df.at[aid, 'rt_num_resolved'] = new_n

    hours = float(meta.get('resolution_time_hours', 0.0))
    pp_df.at[aid, 'rt_total_resolution_hours'] = float(pp_df.at[aid, 'rt_total_resolution_hours']) + hours
    if old_n == 0:
        pp_df.at[aid, 'rt_max_resolution_hours'] = hours
        pp_df.at[aid, 'rt_avg_resolution_hours'] = hours
    else:
        pp_df.at[aid, 'rt_max_resolution_hours'] = max(float(pp_df.at[aid, 'rt_max_resolution_hours']), hours)
        pp_df.at[aid, 'rt_avg_resolution_hours'] = float(pp_df.at[aid, 'rt_total_resolution_hours']) / new_n

    rating = meta.get('satisfaction_rating')
    if rating is not None:
        pp_df.at[aid, 'rt_num_satrat_responses'] = int(pp_df.at[aid, 'rt_num_satrat_responses']) + 1
        if float(rating) < 2.5:
            pp_df.at[aid, 'rt_num_low_satrat'] = int(pp_df.at[aid, 'rt_num_low_satrat']) + 1
        else:
            pp_df.at[aid, 'rt_num_high_satrat'] = int(pp_df.at[aid, 'rt_num_high_satrat']) + 1
    pp_df.at[aid, 'rt_satrat_response_rate'] = int(pp_df.at[aid, 'rt_num_satrat_responses']) / new_n

    days = (now - pd.Timestamp(event['timestamp'])).days
    _update_recency(pp_df, aid, 'rt_days_since_latest_resolution', days, is_first=(old_n == 0))


def _strategy_payment_processed(pp_df: pd.DataFrame, event: dict, now: pd.Timestamp) -> None:
    aid = event['account_id']
    amount = float(event.get('metadata', {}).get('amount', 0.0))
    pp_df.at[aid, 'rt_num_payments_processed'] = int(pp_df.at[aid, 'rt_num_payments_processed']) + 1
    pp_df.at[aid, 'rt_total_payment_amount'] = float(pp_df.at[aid, 'rt_total_payment_amount']) + amount


def _strategy_payment_failed(pp_df: pd.DataFrame, event: dict, now: pd.Timestamp) -> None:
    aid = event['account_id']
    amount = float(event.get('metadata', {}).get('amount', 0.0))
    pp_df.at[aid, 'rt_num_payments_failed'] = int(pp_df.at[aid, 'rt_num_payments_failed']) + 1
    pp_df.at[aid, 'rt_total_failed_amount'] = float(pp_df.at[aid, 'rt_total_failed_amount']) + amount


def _strategy_integration_added(pp_df: pd.DataFrame, event: dict, now: pd.Timestamp) -> None:
    aid = event['account_id']
    pp_df.at[aid, 'integration_count'] = int(pp_df.at[aid, 'integration_count']) + 1


def _strategy_integration_removed(pp_df: pd.DataFrame, event: dict, now: pd.Timestamp) -> None:
    aid = event['account_id']
    # Floor at 0 — a removal event for an already-zero account is a data
    # inconsistency we tolerate rather than fail on.
    pp_df.at[aid, 'integration_count'] = max(0, int(pp_df.at[aid, 'integration_count']) - 1)


def _strategy_seat_added(pp_df: pd.DataFrame, event: dict, now: pd.Timestamp) -> None:
    aid = event['account_id']
    total = int(event.get('metadata', {}).get('total_seats', pp_df.at[aid, 'seats_active']))
    pp_df.at[aid, 'seats_active'] = total
    # Purchases ratchet up only — seats_purchased never shrinks.
    pp_df.at[aid, 'seats_purchased'] = max(int(pp_df.at[aid, 'seats_purchased']), total)


def _strategy_seat_removed(pp_df: pd.DataFrame, event: dict, now: pd.Timestamp) -> None:
    aid = event['account_id']
    total = int(event.get('metadata', {}).get('total_seats', pp_df.at[aid, 'seats_active']))
    pp_df.at[aid, 'seats_active'] = total
    pp_df.at[aid, 'seats_purchased'] = max(int(pp_df.at[aid, 'seats_purchased']), total)


_STRATEGIES: dict[str, Callable[[pd.DataFrame, dict, pd.Timestamp], None]] = {
    'login':                   _strategy_login,
    'logout':                  _strategy_logout,
    'feature_usage':           _strategy_feature_usage,
    'report_generated':        _strategy_report_generated,
    'dashboard_created':       _strategy_dashboard_created,
    'support_ticket_created':  _strategy_support_ticket_created,
    'support_ticket_resolved': _strategy_support_ticket_resolved,
    'payment_processed':       _strategy_payment_processed,
    'payment_failed':          _strategy_payment_failed,
    'integration_added':       _strategy_integration_added,
    'integration_removed':     _strategy_integration_removed,
    'seat_added':              _strategy_seat_added,
    'seat_removed':            _strategy_seat_removed,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_events(pp_df: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    """Fold an event batch into a fresh copy of pp_df.

    Events are deduped by event_id and applied in timestamp order. The window
    anchor ("now") is max(event timestamp) in the batch — used for rt_days_since_*
    recency columns.

    Events whose account_id is not in pp_df.index are skipped with a warning;
    we don't fabricate rows for unknown accounts.

    Returns a new DataFrame. The input pp_df is never mutated.
    """
    if not events:
        return pp_df.copy()

    sorted_events = _dedup_and_sort(events)
    now = pd.Timestamp(sorted_events[-1]['timestamp'])

    out = pp_df.copy()
    known = set(out.index)
    for event in sorted_events:
        _validate_event(event)
        aid = event['account_id']
        if aid not in known:
            logger.warning("apply_events: account_id %r not in pp_df.index, skipping", aid)
            continue
        allowed = EVENT_MAPPING[event['event_type']]
        before = out.loc[aid].copy()
        _STRATEGIES[event['event_type']](out, event, now)
        _validate_postcondition(before, out.loc[aid], allowed, event['event_type'])
    return out
