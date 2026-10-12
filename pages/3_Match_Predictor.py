"""Match probabilities and score estimates for chosen fixtures."""
import os
from datetime import datetime, timezone

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from pitchiq.data import validate_matches
from pitchiq.predictions import (
    history_snapshots, make_feature_row, probabilities_from_model,
    rolling_form, scoreline_distribution,
)
from pitchiq.real_matches import CURRENT_SEASON, load_real_fixtures, load_real_matches
from pitchiq.real_models import load_real_models
from pitchiq.ui import PALETTE, configure_page, footer, page_header

configure_page("Match predictor")
page_header("Match predictor", "Historical team form, upcoming fixtures and experimental score estimates.", "Predict / Match outcomes")

# Offline demo mode shouldn't try downloading match data.
source_options = ["Real OpenFootball results", "Fictional demo"]
initial = 1 if os.getenv("PITCHIQ_DATA_MODE") == "demo" else 0
source = st.radio("Match data source", source_options, horizontal=True, index=initial, key="predict_source_2026")
real = source == source_options[0]
st.caption("Real mode uses OpenFootball completed results and unplayed fixtures. Models run in memory; no player or shot datasets are downloaded." if real else "Fictional clubs and trained-on-fictional-data models for demonstration.")

try:
    if real:
        matches = load_real_matches()
    else:
        from pitchiq.demo import matches as demo_matches, model as demo_model
        matches = validate_matches(demo_matches())
except Exception as error:
    st.error("Cannot load match results. Try Fictional demo or check OpenFootball access.")
    with st.expander("Error details"):
        st.code(str(error))
    st.stop()

seasons = sorted((s for s in matches.Season.unique() if not real or s >= "2022-23"), reverse=True)
season = st.selectbox("Season", seasons, index=0, key="predict_season_2026")
season_matches = matches[matches.Season == season]
competitions = sorted(season_matches.Competition.unique())
if not len(competitions):
    st.warning("No competitions found in this season")
    st.stop()
competition = st.selectbox("Competition", competitions, index=next((i for i, c in enumerate(competitions) if c == "Premier League"), 0), key="predict_competition_2026")
team_rows = season_matches[season_matches.Competition == competition]
teams = sorted(set(team_rows.HomeTeam) | set(team_rows.AwayTeam))
if len(teams) < 2:
    st.warning("Not enough teams in this competition")
    st.stop()

# Keep scheduled matches separate from completed results.
fixtures = pd.DataFrame()
today = pd.Timestamp(datetime.now(timezone.utc).date())
if real and season == CURRENT_SEASON:
    try:
        fixtures = load_real_fixtures(season)
        fixtures = fixtures[(fixtures.Competition == competition) & (fixtures.Date >= today)].copy()
        fixtures = fixtures.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    except Exception as error:
        st.warning(f"Upcoming fixture list is unavailable ({error}); custom historical matchup remains available.")

upcoming = len(fixtures) > 0
mode = "Custom hypothetical matchup"
if upcoming:
    mode = st.radio("Prediction type", ["Upcoming scheduled fixture", "Custom hypothetical matchup"], horizontal=True, key="predict_type_2026")
if mode == "Upcoming scheduled fixture":
    lookup = list(range(len(fixtures)))
    index = st.selectbox(
        "Upcoming fixture", lookup,
        format_func=lambda i: f"{fixtures.iloc[i].Date.strftime('%d %b %Y')} · {fixtures.iloc[i].HomeTeam} vs {fixtures.iloc[i].AwayTeam}",
        key="predict_fixture_2026",
    )
    picked = fixtures.iloc[index]
    home, away = picked.HomeTeam, picked.AwayTeam
    as_of = pd.Timestamp(picked.Date)
    st.caption(f"Scheduled date: {as_of.date()} · OpenFootball may revise kickoff times or fixtures.")
else:
    a, b = st.columns(2)
    with a:
        home = st.selectbox("Home team", teams, key="predict_home_2026")
    with b:
        away = st.selectbox("Away team", [t for t in teams if t != home], key="predict_away_2026")
    as_of = today + pd.Timedelta(days=1) if real and season == CURRENT_SEASON else None

try:
    if real:
        # Use earlier seasons for training, not matches we're predicting.
        artifact = load_real_models(matches, target_season=season)
        v3 = artifact
        st.warning(
            f"**Experimental, not a guaranteed forecast.** Models train through {artifact['training_through']} "
            f"and use completed {season} results only for team form. "
            "Lineups, injuries, transfers and player availability are not modeled."
        )
    else:
        artifact = demo_model("match_predictor_v2.joblib")
        v3 = demo_model("match_predictor_v3.joblib")
    features = list(artifact["features"])
    if features != list(v3["features"]):
        raise ValueError("V2 and V3 feature schema mismatch")
    @st.cache_data(show_spinner="Calculating historic team form…")
    def cached_history(df, selected_competition, selected_season, rec, k, adv):
        return history_snapshots(df, selected_competition, selected_season, rec, k, adv)
    recent = int(artifact.get("recent_matches", 5))
    minimum = int(artifact.get("minimum_history", 3))
    history, elo, last, last_date = cached_history(matches, competition, season, recent,
        float(artifact.get("elo_k_factor", 20)), float(artifact.get("home_elo_advantage", 60)))
    if last_date is None or min(len(history.get(home, [])), len(history.get(away, []))) < minimum:
        st.warning("Not enough historical matches for this matchup. Choose other teams or another season.")
        st.stop()
    if as_of is None:
        as_of = last_date + pd.Timedelta(days=7)
    as_of = max(as_of, last_date + pd.Timedelta(days=1))
    input_frame = make_feature_row(home, away, history, elo, last, as_of, recent, features)
    probs = probabilities_from_model(artifact["model"], input_frame)
    hg = float(np.clip(v3["poisson_home_model"].predict(input_frame)[0], 0.05, 8))
    ag = float(np.clip(v3["poisson_away_model"].predict(input_frame)[0], 0.05, 8))
except Exception as error:
    st.error("Could not train or evaluate this matchup. Switch to Fictional demo to check the UI.")
    with st.expander("Error details"):
        st.code(str(error))
    st.stop()

st.caption(f"{season} snapshot through {last_date.date()} · estimated matchup date {as_of.date()}")
st.subheader("Recent team performance")
for col, team in zip(st.columns(2), (home, away)):
    points, goals_for, goals_against, wins = rolling_form(history[team], recent)
    with col:
        st.markdown(f"**{team}**")
        x, y = st.columns(2)
        x.metric("Points / game", f"{points:.2f}")
        y.metric("Win rate", f"{wins:.0%}")
        x.metric("Scored / game", f"{goals_for:.2f}")
        y.metric("Conceded / game", f"{goals_against:.2f}")
        st.caption(f"Elo: {elo.get(team, 1500):.0f}")
st.divider()
st.header("Match prediction")
for col, title, cls in zip(st.columns(3), (f"{home} win", "Draw", f"{away} win"), ("H", "D", "A")):
    col.metric(title, f"{probs[cls]:.1%}")
chart = pd.DataFrame({"Outcome": [home + " win", "Draw", away + " win"], "Probability": [100 * probs[c] for c in ("H", "D", "A")]})
st.altair_chart(alt.Chart(chart).mark_bar(cornerRadiusEnd=6).encode(
    x=alt.X("Probability:Q", title="Estimated probability (%)", scale=alt.Scale(domain=[0, 100])),
    y=alt.Y("Outcome:N", title=None, sort=chart.Outcome.tolist()),
    color=alt.Color("Outcome:N", legend=None, scale=alt.Scale(domain=chart.Outcome.tolist(), range=PALETTE[:3])),
    tooltip=["Outcome:N", alt.Tooltip("Probability:Q", format=".1f")],
).properties(height=175), width="stretch")
cls = max(probs, key=probs.get)
outcome_text = {"H": home + " win", "D": "Draw", "A": away + " win"}[cls]
st.info(f"Highest estimated outcome: **{outcome_text}**")
st.divider()
st.header("Expected goals and scorelines")
a, b = st.columns(2)
a.metric(f"{home} estimated goals", f"{hg:.2f}")
b.metric(f"{away} estimated goals", f"{ag:.2f}")
st.caption("Goal estimates come from a Poisson model, not shot-based xG. They are uncertain.")
scores = pd.DataFrame([{"Score": f"{home} {gh}–{ga} {away}", "Probability": p * 100} for gh, ga, p in scoreline_distribution(hg, ag)])
st.dataframe(scores.style.format({"Probability": "{:.1f}%"}), hide_index=True, width="stretch")
st.caption("Outcome and exact-score probabilities use different experimental models and do not necessarily agree.")
with st.expander("Model validation"):
    if real:
        report = artifact["test_metrics"]
        st.write(f"Before training for {season}, we tested on {report['test_season']} using older seasons only: {report['test_matches']:,} matches, accuracy {report['accuracy']:.2%}, log loss {report['log_loss']:.4f}.")
        st.caption("Results on an earlier season do not validate accuracy in 2026–27. Models are automatically retrained from latest completed source data as cache updates.")
    else:
        st.write("The fictional model was trained and evaluated on generated data only.")
footer("OpenFootball CC0 match records · Experimental in-memory models" if real else "Original fictional demonstration")
