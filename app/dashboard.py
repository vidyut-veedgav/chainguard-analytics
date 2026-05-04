"""ChurnGuard dashboard + chat — single-view Streamlit app.

Run:
    uv run streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts app/ on sys.path, not the project root — make `src` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from src import api
from src.agent import ask


# ---------------------------------------------------------------------------
# Cached wrappers — Streamlit reruns the whole script per interaction; these
# skip re-serialization on top of api.py's own lru_cache.
# ---------------------------------------------------------------------------

@st.cache_data
def _portfolio_summary() -> dict:
    return api.portfolio_summary()


@st.cache_data
def _feature_importance(top_k: int = 10) -> list[dict]:
    return api.feature_importance(top_k=top_k)


@st.cache_data
def _probability_distribution(bins: int = 20) -> dict:
    return api.probability_distribution(bins=bins)


@st.cache_data
def _account_ids() -> list[str]:
    return api._load_scoring_frame().index.astype(str).tolist()


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(layout='wide', page_title='ChurnGuard')

if 'chat' not in st.session_state:
    st.session_state.chat = None
    st.session_state.messages = []

dash_col, chat_col = st.columns([2, 1], gap='large')


# ---------------------------------------------------------------------------
# Dashboard column
# ---------------------------------------------------------------------------

with dash_col:
    st.title('ChurnGuard Analytics')

    summary = _portfolio_summary()
    mm = summary['model_metrics']

    # Hero tile — accounts at risk dominates the page
    st.markdown('### Accounts at Risk')
    st.metric(
        label=f"Active accounts above probability threshold ({summary['threshold']:.2f})",
        value=summary['n_high_risk'],
        delta=f"{summary['pct_high_risk']:.1%} of active book",
        delta_color='off',
    )

    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric('Total accounts', summary['total_accounts'])
    c2.metric('Active', summary['n_active'])
    c3.metric('Avg probability', f"{summary['avg_probability']:.2%}")

    # Population base rate for context — `avg_probability` above is over the
    # active book only (churners excluded), so it sits well below the full-book
    # churn rate by construction. Surfacing both prevents the "1.5% looks
    # broken" misread.
    base_rate = summary['n_churned'] / summary['total_accounts'] if summary['total_accounts'] else 0.0
    c3.caption(f"Active accounts only. Population churn rate: {base_rate:.1%}")

    c4, c5, c6 = st.columns(3)
    c4.metric('Precision', f"{mm['precision']:.3f}")
    c5.metric('Recall', f"{mm['recall']:.3f}")
    c6.metric('F1', f"{mm['f1']:.3f}")

    st.divider()

    # Probability distribution (active book) — log y-axis. The active book is
    # heavily right-skewed (most accounts score near zero by design), so a
    # linear axis flattens the high-risk tail into invisibility.
    st.subheader('Distribution of churn probabilities')
    st.caption('Active accounts only, right tail is actionable.')
    dist = _probability_distribution(bins=20)
    hist_df = pd.DataFrame({
        'probability_bin': [f'{e:.2f}' for e in dist['bin_edges'][:-1]],
        'count':           dist['counts'],
    })
    chart = (
        alt.Chart(hist_df)
        .mark_bar()
        .encode(
            x=alt.X('probability_bin:N', title='probability bin', sort=None),
            # symlog handles the zero-count bins cleanly; pure log would error.
            y=alt.Y('count:Q', scale=alt.Scale(type='symlog'), title='count (log)'),
            tooltip=['probability_bin', 'count'],
        )
    )
    threshold_rule = (
        alt.Chart(pd.DataFrame({'bin': [f"{dist['threshold']:.2f}"]}))
        .mark_rule(color='red', strokeDash=[4, 4])
        .encode(x='bin:N')
    )
    st.altair_chart(chart + threshold_rule, width='stretch')

    st.divider()

    # Top-N risk + global feature importance, side by side
    left, right = st.columns(2)

    with left:
        st.subheader('Top 10 highest-risk accounts')
        top = api.top_risk_accounts(n=10)
        st.dataframe(pd.DataFrame(top), hide_index=True, width='stretch')

    with right:
        st.subheader('Top 10 features (global importance)')
        fi = _feature_importance(top_k=10)
        fi_df = pd.DataFrame(fi).set_index('feature')[['importance']]
        st.bar_chart(fi_df)

    st.divider()

    # Per-account drill-down
    st.subheader('Account drill-down')
    selected = st.selectbox('Account', options=_account_ids(), index=0)
    explanation = api.explain_account(selected, top_k=5)

    e1, e2, e3 = st.columns(3)
    e1.metric('Probability', f"{explanation['probability']:.3f}")
    e2.metric('Predicted', explanation['predicted_label'])
    e3.metric('Actual status', explanation['actual_status'])

    drivers_col, protectors_col = st.columns(2)
    with drivers_col:
        st.markdown('**Top drivers (push toward churn)**')
        st.dataframe(pd.DataFrame(explanation['top_drivers']), hide_index=True, width='stretch')
    with protectors_col:
        st.markdown('**Top protectors (push toward retention)**')
        st.dataframe(pd.DataFrame(explanation['top_protectors']), hide_index=True, width='stretch')


# ---------------------------------------------------------------------------
# Chat column
# ---------------------------------------------------------------------------

with chat_col:
    st.title('Ask ChurnGuard')
    st.caption('Grounded in the same model. Try: "top 5 churn risks" or "why is ACC000123 risky?"')

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    user_text = st.chat_input('Ask about churn risk...')
    if user_text:
        st.session_state.messages.append({'role': 'user', 'content': user_text})
        with st.chat_message('user'):
            st.markdown(user_text)

        with st.chat_message('assistant'):
            try:
                reply, chat = ask(user_text, st.session_state.chat)
                st.session_state.chat = chat
            except Exception as e:
                reply = f"Sorry — I hit an error: `{type(e).__name__}: {e}`"
            st.markdown(reply)

        st.session_state.messages.append({'role': 'assistant', 'content': reply})
