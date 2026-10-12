"""Domestic league projections and an experimental knockout simulator."""
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from pitchiq.data import validate_matches
from pitchiq.real_matches import CURRENT_SEASON, load_real_fixtures, load_real_matches
from pitchiq.real_models import load_real_models
from pitchiq.simulation import current_table, league_simulate, predict_many, snapshot, ucl_simulate
from pitchiq.ui import configure_page, download_csv, footer, page_header, probability_chart

configure_page("League simulator")
page_header("League & UCL simulator", "2026–27 league projections and historical Champions League what-if brackets.", "Simulate / League outcomes")
source = st.radio(
    "Match data source", ["Real OpenFootball results", "Fictional demo"],
    horizontal=True, index=1 if os.getenv("PITCHIQ_DATA_MODE") == "demo" else 0,
    key="league_source_2026",
)
real = source.startswith("Real")
try:
    if real:
        matches = load_real_matches()
    else:
        from pitchiq.demo import matches as demo_matches, model as demo_model
        matches = validate_matches(demo_matches())
except Exception as error:
    st.error("Could not load match results. Try Fictional demo, or check source availability.")
    with st.expander("Error details"):
        st.code(str(error))
    st.stop()

available = sorted((s for s in matches.Season.unique() if not real or s >= "2022-23"), reverse=True)
season = st.selectbox("Season", available, index=0, key="league_season_2026")
season_data = matches[matches.Season == season]
competitions = sorted(season_data.Competition.unique())
if not competitions:
    st.warning(f"No completed matches in {season}")
    st.stop()
competition = st.selectbox(
    "Competition", competitions,
    index=next((i for i, label in enumerate(competitions) if label == "Premier League" or label == "Demo Premier League"), 0),
    key="league_competition_2026",
)
comp = season_data[season_data.Competition == competition].sort_values("Date").reset_index(drop=True)
teams = sorted(set(comp.HomeTeam) | set(comp.AwayTeam))
if not teams:
    st.warning("No teams in this competition")
    st.stop()

try:
    if real:
        model = load_real_models(matches, target_season=season)
        st.caption(f"Models trained on completed results through {model['training_through']}; {season} outcomes are not training targets.")
    else:
        model = demo_model("match_predictor_v3.joblib")
except Exception as error:
    st.error("Could not prepare models for this season")
    with st.expander("Error details"):
        st.code(str(error))
    st.stop()

is_cup = "champions" in competition.lower() or "ucl" in competition.lower()
if is_cup:
    st.subheader("🏆 Hypothetical Champions League knockout")
    st.info("This is a **hypothetical 16-team cup**, not the official group/league-stage draw, bracket or qualification probabilities. 2026–27 UCL fixtures have not been integrated.")
    entrants = st.multiselect("Select exactly 16 entrants", teams, default=teams[:16], max_selections=16, key="ucl_entrants_2026")
    runs = st.slider("Cup simulations", 200, 5000, 1000, step=200, key="ucl_runs_2026")
    seed = st.number_input("Random seed", 0, 999999, 42, key="ucl_seed_2026")
    if len(entrants) != 16:
        st.warning("Choose exactly 16 teams")
        st.stop()
    if st.button("🏆 Simulate knockout cup", type="primary"):
        cutoff = comp.Date.min()
        history, elo, last = snapshot(matches, competition, cutoff)
        with st.spinner("Running simulations…"):
            report, bracket = ucl_simulate(entrants, cutoff, history, elo, last, model, runs, int(seed))
        st.session_state["ucl_2026_run"] = ((source, season, competition, tuple(entrants), runs, int(seed)), report, bracket)
    saved = st.session_state.get("ucl_2026_run")
    key = (source, season, competition, tuple(entrants), runs, int(seed))
    if saved and saved[0] == key:
        report, bracket = saved[1:]
        st.subheader("Simulated trophy probabilities")
        st.dataframe(report, hide_index=True, width="stretch")
        probability_chart(report, "Win trophy %")
        download_csv(report, "pitchiq_ucl_projection.csv")
        st.subheader("Example fictional bracket")
        for stage, ties in bracket:
            with st.expander(stage, expanded=stage == "Final"):
                for home, away, winner, hg, ag in ties:
                    st.write(f"{home} **{hg}–{ag}** {away} · Advances: **{winner}**")
else:
    st.subheader("📊 League standings and Monte Carlo projections")
    if real and season == CURRENT_SEASON:
        # Split played matches from fixtures before projecting the table.
        as_of = pd.Timestamp(datetime.now(timezone.utc).date()) + pd.Timedelta(days=1)
        played = comp.copy()
        all_planned = load_real_fixtures(season)
        remaining = all_planned[all_planned.Competition == competition].copy()
        # Don't treat future fixtures as results.
        cutoff = as_of
        st.caption("2026–27: uses every verified completed match available through today. The remaining schedule is read directly from OpenFootball and may change.")
    else:
        dates = sorted(comp.Date.unique())
        pos = st.slider("Historical cutoff (completed match dates)", 0, len(dates), len(dates) // 2, key="league_cutoff_2026")
        cutoff = pd.Timestamp(dates[pos]) if pos < len(dates) else pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)
        played = comp[comp.Date < cutoff].copy()
        remaining = comp[comp.Date >= cutoff].copy()
        st.caption("Historical what-if: results on and after the cutoff are hidden from the starting standings and team form.")

    # Include newly promoted teams even when they have little history.
    teams = sorted(set(teams) | set(remaining.HomeTeam) | set(remaining.AwayTeam))
    scheduled = len(played) + len(remaining)
    expected_schedule = len(teams) * (len(teams) - 1)
    is_complete_schedule = expected_schedule == scheduled
    if real and season == CURRENT_SEASON and not is_complete_schedule:
        st.warning(
            f"OpenFootball currently lists {scheduled:,} completed/pending fixtures, versus "
            f"{expected_schedule:,} expected in a full double round-robin. "
            "Any result would project only listed fixtures, **not a complete final league table**."
        )
    a, b, c = st.columns(3)
    a.metric("Completed games", f"{len(played):,}")
    b.metric("Pending fixtures", f"{len(remaining):,}")
    c.metric("Projection as of", (cutoff - pd.Timedelta(days=1)).strftime("%d %b %Y") if real and season == CURRENT_SEASON else cutoff.strftime("%d %b %Y"))
    st.subheader("Actual standings at cutoff")
    st.dataframe(current_table(played, teams), hide_index=True, width="stretch")
    runs = st.slider("League simulations", 200, 10000, 2000, step=200, key="league_runs_2026")
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, key="league_seed_2026")
    key = (source, season, competition, cutoff.isoformat(), len(played), len(remaining), runs, int(seed))
    if st.button("🎲 Simulate remaining fixtures", type="primary"):
        history, elo, last = snapshot(matches, competition, cutoff)
        # Handle postponed fixtures as what-if games, not past results.
        fixtures = [(r.HomeTeam, r.AwayTeam, max(pd.Timestamp(r.Date), cutoff)) for r in remaining.itertuples(index=False)]
        with st.spinner("Simulating remaining fixtures…"):
            estimates = predict_many(fixtures, history, elo, last, model)
            report = league_simulate(played, remaining, teams, estimates, runs, int(seed))
        st.session_state["league_2026_run"] = (key, report)
    saved = st.session_state.get("league_2026_run")
    if saved and saved[0] == key:
        report = saved[1]
        st.subheader("Projected final standings" if is_complete_schedule else "Projected points for listed fixtures only")
        st.dataframe(report, hide_index=True, width="stretch")
        probability_chart(report, "Title %")
        download_csv(report, "pitchiq_league_projection.csv")
        st.caption("Model simplifications: frozen team strengths at cutoff, independent Poisson goals, simplified tie-breakers. Top-four/bottom-three represent finishing positions, not guaranteed qualification or relegation. Newly promoted teams have limited history.")

footer("OpenFootball CC0 historical results, published schedules, experimental models" if real else "Original fictional demo")
