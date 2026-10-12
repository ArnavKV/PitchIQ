"""Overview with real dataset coverage and links to every tool."""
import streamlit as st

from pitchiq.data import load_matches, using_demo
from pitchiq.ui import demo_notice, footer

mode_label = "FICTIONAL DEMO + OPTIONAL HISTORICAL MATCHES" if using_demo() else "HISTORICAL DATA &nbsp; / &nbsp; EXPERIMENTAL ML"
st.markdown('''<div class="iq-eyebrow">The football analytics workspace</div>
<div class="iq-hero"><div><div class="iq-eyebrow">A different view of the game</div>
<h1>Follow the numbers.<br><span class="iq-accent">Understand the game.</span></h1>
<p>Compare the players. Explore the results. Play out the possibilities.
Pick a tool and explore the numbers.</p>
<span class="iq-pill">__MODE_LABEL__</span></div>
<div class="iq-field" aria-hidden="true"><div class="iq-circle"></div><div class="iq-box left"></div>
<div class="iq-box right"></div><i class="iq-dot one"></i><i class="iq-dot two"></i>
<i class="iq-dot three"></i><i class="iq-dot four"></i><div class="iq-field-label">THE GAME, IN FOCUS</div></div></div>'''.replace("__MODE_LABEL__", mode_label), unsafe_allow_html=True)
demo_notice()

try:
    matches = load_matches()
except (OSError, ValueError) as exc:
    st.info("Match coverage will appear once the local dataset is ready. Run `python load_matches.py` to prepare it.")
    with st.expander("Dataset details"):
        st.write(str(exc))
else:
    seasons = sorted(matches["Season"].unique())
    with st.container(key="overview_metrics"):
        a, b, c, d = st.columns(4)
        a.metric("Demo matches" if using_demo() else "Recorded matches", f"{len(matches):,}")
        b.metric("Competitions", matches["Competition"].nunique())
        c.metric("Seasons of results", len(seasons))
        d.metric("Analytics tools", "6")
    if seasons:
        label = "Fictional season labels" if using_demo() else "Match coverage"
        st.caption(f"{label}: {seasons[0]} to {seasons[-1]}. Match tools have a separate real OpenFootball mode; homepage counts describe {'fictional' if using_demo() else 'historical'} data.")

st.markdown("## Find your next insight")
st.caption("Start with a question. Choose a tool to explore it.")
tools = [
    ("01", "Player comparison", "Who has the edge? Compare scoring, creativity, and per-90 performance across recorded seasons.", "pages/1_Player_Comparison.py", "Compare players"),
    ("02", "Match explorer", "What does the record say? Filter European results by season, competition, and team.", "pages/2_Match_Explorer.py", "Explore matches"),
    ("03", "Match predictor", "How might they match up? Explore win probabilities and expected scores from historical team strength.", "pages/3_Match_Predictor.py", "Build a matchup"),
    ("04", "xG predictor", "How good is that chance? Place a shot on the pitch and explore its estimated scoring probability.", "pages/4_xG_Predictor.py", "Explore a chance"),
    ("05", "Shot simulator", "Can you find the net? Set up a shooter and keeper, aim your attempt, and watch it play out.", "pages/5_Shot_Simulator.py", "Take a shot"),
    ("06", "League simulator", "Who could lift the trophy? Simulate a league run-in or a hypothetical Champions League knockout cup.", "pages/6_League_Simulator.py", "Simulate a season"),
]
for start in (0, 3):
    for column, (number, title, description, path, label) in zip(st.columns(3), tools[start:start + 3]):
        with column, st.container(border=True):
            st.markdown(f'<div class="iq-tool-icon">{number}</div>', unsafe_allow_html=True)
            st.subheader(title)
            st.markdown(f'<p class="iq-card-description">{description}</p>', unsafe_allow_html=True)
            st.page_link(path, label=label, icon=":material/arrow_forward:")

footer("Optional historical match results: OpenFootball CC0 · Other public tools: fictional demo.")
