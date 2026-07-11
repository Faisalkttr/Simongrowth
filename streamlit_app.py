"""
Streamlit entry point for the Sovereign Expectations Engine.

DEPLOYMENT NOTE:
On Streamlit Community Cloud, set this file (streamlit_app.py) as the
"Main file path" in the app settings. app1.py contains only the engine
class and has no Streamlit UI code of its own, which is why the
previously-deployed app rendered a blank page.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app1 import SovereignExpectationsEngine

st.set_page_config(
    page_title="Sovereign Expectations Engine",
    page_icon="📊",
    layout="wide",
)

# ----------------------------------------------------------------------
# Cached data pulls — avoids re-hitting yfinance on every widget interaction
# ----------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def run_engine(ticker: str, is_core: bool):
    """
    Runs the full engine pipeline for a ticker and returns both the
    computed metrics dict and the underlying df_data (for charting).
    Cached for an hour per (ticker, is_core) combination so repeated
    interactions with the page don't re-download data every time.
    """
    engine = SovereignExpectationsEngine(ticker, is_core=is_core)
    df_data = engine.hydrate_standalone_data()
    metrics = engine.execute(df_data)
    return metrics, df_data


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------

st.sidebar.title("📊 Sovereign Expectations Engine")
st.sidebar.caption("Implied market expectations & priced-for-perfection risk")

ticker_input = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
is_core = st.sidebar.checkbox(
    "Core holding",
    value=False,
    help="Core holdings are anchored to the 75th percentile historical "
         "multiple (scarcity premium tolerated). Non-core holdings are "
         "anchored to the historical median multiple.",
)
run_clicked = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

with st.sidebar.expander("Status legend"):
    for label, description in SovereignExpectationsEngine.EXPECTATIONS_STATUS_LEGEND.items():
        st.markdown(f"**{label}**  \n{description}")

# ----------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------

st.title("Sovereign Expectations Engine")

if not ticker_input:
    st.info("Enter a ticker in the sidebar and click **Run Analysis** to begin.")
    st.stop()

if not run_clicked and "last_result" not in st.session_state:
    st.info("Click **Run Analysis** in the sidebar to fetch data and compute metrics.")
    st.stop()

if run_clicked:
    try:
        with st.spinner(f"Fetching data and computing expectations for {ticker_input}..."):
            metrics, df_data = run_engine(ticker_input, is_core)
        st.session_state["last_result"] = (ticker_input, metrics, df_data)
    except ValueError as e:
        st.error(f"Could not complete analysis: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while analyzing {ticker_input}: {e}")
        with st.expander("Show technical details"):
            st.exception(e)
        st.stop()

if "last_result" not in st.session_state:
    st.stop()

shown_ticker, metrics, df_data = st.session_state["last_result"]

# ----------------------------------------------------------------------
# Headline status
# ----------------------------------------------------------------------

st.subheader(f"{shown_ticker} — {metrics['Expectations Classification']}")

score = metrics["Expectations Burden Score"]
score_col, conf_col = st.columns([2, 1])
with score_col:
    st.progress(min(max(score / 100, 0.0), 1.0))
    st.caption(f"Expectations Burden Score: {score:.1f} / 100")
with conf_col:
    st.caption(
        f"Forward estimate confidence: **{metrics['Forward Confidence']}**  \n"
        f"Valuation anchor confidence: **{metrics['Valuation Anchor Confidence']}** "
        f"({metrics['Valuation Anchor Observation Count']} observations, "
        f"{metrics['Revenue Data Cadence']} revenue cadence)"
    )

if metrics["Valuation Anchor Confidence"] in ("low", "none"):
    st.warning(
        "Historical P/S anchors are based on limited or reconstructed data "
        "for this ticker. Treat the required-growth and score figures below "
        "as directional, not precise."
    )
if metrics["Forward Confidence"].startswith("Low"):
    st.warning(
        "No analyst estimate or usable historical growth trend was found. "
        "Forward growth has been neutralized to 0% as a conservative fallback."
    )

st.divider()

# ----------------------------------------------------------------------
# Key metrics
# ----------------------------------------------------------------------


def fmt_pct(x):
    return f"{x:.1%}" if x == x and np.isfinite(x) else "N/A"


def fmt_num(x, prefix="", suffix=""):
    return f"{prefix}{x:,.2f}{suffix}" if x == x and np.isfinite(x) else "N/A"


def fmt_money(x):
    if x != x or not np.isfinite(x):
        return "N/A"
    for divisor, label in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(x) >= divisor:
            return f"${x / divisor:,.2f}{label}"
    return f"${x:,.0f}"


c1, c2, c3, c4 = st.columns(4)
c1.metric("Current P/S", fmt_num(metrics["Current P/S"]))
c2.metric(f"{metrics['Target Multiple Label']}", fmt_num(metrics["Target Multiple Value"]))
c3.metric("Forward Revenue Growth (est.)", fmt_pct(metrics["Forward Revenue Growth Estimate"]))
c4.metric("Required Revenue Growth", fmt_pct(metrics["Required Revenue Growth"]))

c5, c6, c7, c8 = st.columns(4)
c5.metric("Current Market Cap", fmt_money(metrics["Current Market Cap"]))
c6.metric("Current Revenue (TTM)", fmt_money(metrics["Current Revenue TTM"]))
c7.metric("Growth Gap", fmt_pct(metrics["Growth Gap"]))
years = metrics["Years to Normalise Multiple"]
years_display = "Already normalised" if years == 0 else ("∞ (no growth)" if years == np.inf else (f"{years:.1f} yrs" if years == years else "N/A"))
c8.metric("Years to Normalise Multiple", years_display)

st.caption(f"Forward source: {metrics['Forward Source Pipeline']}")

st.divider()

# ----------------------------------------------------------------------
# P/S history chart
# ----------------------------------------------------------------------

st.subheader("Historical P/S Ratio")

ps_series = df_data["PS_Ratio"].replace([np.inf, -np.inf], np.nan).dropna()

if ps_series.empty:
    st.info("No valid historical P/S series is available to chart for this ticker.")
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ps_series.index, y=ps_series.values,
        mode="lines", name="P/S Ratio", line=dict(width=2),
    ))
    fig.add_hline(
        y=metrics["Historical Median P/S"], line_dash="dot",
        annotation_text="Median", annotation_position="top left",
    )
    fig.add_hline(
        y=metrics["Historical 75th Percentile P/S"], line_dash="dash",
        annotation_text="75th pct", annotation_position="top left",
    )
    fig.add_hline(
        y=metrics["Historical 90th Percentile P/S"], line_dash="dashdot",
        annotation_text="90th pct", annotation_position="top left",
    )
    fig.add_hline(
        y=metrics["Current P/S"], line_color="red",
        annotation_text="Current", annotation_position="bottom left",
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="P/S Ratio",
        xaxis_title=None,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------------------------
# Raw metrics / data
# ----------------------------------------------------------------------

with st.expander("Raw metrics"):
    metrics_df = pd.DataFrame(
        [(k, v) for k, v in metrics.items()],
        columns=["Metric", "Value"],
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

with st.expander("Underlying price / revenue data (tail)"):
    st.dataframe(df_data.tail(50), use_container_width=True)

st.caption(
    "Data sourced from Yahoo Finance via yfinance. Estimates and classifications "
    "are for informational purposes only and are not investment advice."
)
