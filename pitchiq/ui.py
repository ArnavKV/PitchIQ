"""Small presentation helpers shared by every Streamlit page."""
from html import escape
from pathlib import Path

import altair as alt
import streamlit as st

from pitchiq.data import using_demo

ROOT = Path(__file__).resolve().parent.parent
PALETTE = ["#177653", "#d5a846", "#6795ad", "#92b886", "#aa82a2", "#617f75"]


@alt.theme.register("pitchiq", enable=True)
def pitchiq_theme():
    return alt.theme.ThemeConfig({"config": {
        "range": {"category": PALETTE}, "view": {"stroke": None},
        "axis": {"labelColor": "#64746b", "titleColor": "#64746b", "gridColor": "#e8eee7", "domain": False},
        "background": "transparent", "font": "sans-serif",
    }})


def configure_page(title):
    st.set_page_config(page_title=f"{title} · PitchIQ", page_icon="⚽", layout="wide")
    st.logo(str(ROOT / "assets/logo.svg"), size="large", icon_image=str(ROOT / "assets/mark.svg"))
    st.markdown(f"<style>{(ROOT / 'assets/style.css').read_text()}</style>", unsafe_allow_html=True)


def page_header(title, description, eyebrow="Football intelligence"):
    st.markdown(
        f'<div class="iq-eyebrow">{escape(eyebrow)}</div>'
        f'<h1 class="iq-page-title">{escape(title)}</h1>'
        f'<p class="iq-description">{escape(description)}</p>',
        unsafe_allow_html=True,
    )
    demo_notice()


def demo_notice():
    if using_demo():
        st.info(
            "Fictional demo: the Player Comparison, xG Predictor and Shot Simulator "
            "use generated data and models. Match pages offer optional real OpenFootball "
            "results and experimental models. Choose the source on each match page.",
            icon=":material/info:",
        )

def footer(source):
    if using_demo():
        source = "PitchIQ public portfolio · Match tools: optional OpenFootball CC0 · Other tools: fictional demo · Source code: MIT License"
    st.markdown(f'<div class="iq-footer">PitchIQ · {escape(source)}</div>', unsafe_allow_html=True)


def download_csv(frame, filename, label="Download results"):
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"), filename, "text/csv", icon=":material/download:")


def probability_chart(frame, probability_column):
    """A readable horizontal chart; all teams remain available in the report."""
    leaders = frame.nlargest(12, probability_column)
    chart = alt.Chart(leaders).mark_bar(color=PALETTE[0], cornerRadiusEnd=5).encode(
        x=alt.X(f"{probability_column}:Q", title="Estimated probability (%)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Team:N", title=None, sort="-x"),
        tooltip=["Team:N", alt.Tooltip(f"{probability_column}:Q", format=".1f")],
    ).properties(height=max(160, len(leaders) * 28))
    st.altair_chart(chart, width="stretch")
    if len(frame) > len(leaders):
        st.caption("Showing the 12 highest title probabilities. The table includes every team.")
