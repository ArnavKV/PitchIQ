"""Streamlit navigation for the six PitchIQ pages."""
import streamlit as st

from pitchiq.ui import configure_page
from pitchiq.data import using_demo

configure_page("Overview")
navigation = st.navigation({
    "Workspace": [st.Page("views/home.py", title="Overview", icon=":material/grid_view:", default=True)],
    "Explore": [
        st.Page("pages/1_Player_Comparison.py", title="Player comparison", icon=":material/group:", url_path="Player_Comparison"),
        st.Page("pages/2_Match_Explorer.py", title="Match explorer", icon=":material/query_stats:", url_path="Match_Explorer"),
    ],
    "Predict & simulate": [
        st.Page("pages/3_Match_Predictor.py", title="Match predictor", icon=":material/auto_graph:", url_path="Match_Predictor"),
        st.Page("pages/4_xG_Predictor.py", title="xG predictor", icon=":material/my_location:", url_path="xG_Predictor"),
        st.Page("pages/5_Shot_Simulator.py", title="Shot simulator", icon=":material/sports_soccer:", url_path="Shot_Simulator"),
        st.Page("pages/6_League_Simulator.py", title="League simulator", icon=":material/emoji_events:", url_path="League_Simulator"),
    ],
}, expanded=True)
with st.sidebar:
    mode_label = "Match tools: historical OpenFootball opt-in · Players/shots: fictional demo" if using_demo() else "Historical data · Experimental models"
    st.markdown(f'<div class="iq-sidebar-note">FOOTBALL DATA PROJECT<br>Stats, models, and experiments.<br><br>{mode_label}</div>', unsafe_allow_html=True)
navigation.run()
