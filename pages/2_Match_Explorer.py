"""PitchIQ Match Explorer: fictional demo or real OpenFootball historical results.

OpenFootball league JSON and Champions League text are published as CC0.
Data is fetched on demand for the selected season; no datasets are bundled.
"""
import json
import os
import re
import ssl
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import altair as alt
import certifi
import pandas as pd
import streamlit as st

from pitchiq.data import load_matches, validate_matches
from pitchiq.ui import PALETTE, configure_page, download_csv, footer, page_header

configure_page("Match explorer")
page_header(
    "Match explorer",
    "Explore fictional demo matches or historical results from OpenFootball.",
    "Explore / Match results",
)

SEASONS = ["2026-27", "2025-26", "2024-25", "2023-24", "2022-23", "2021-22", "2020-21"]
LEAGUES = {
    "en.1.json": "Premier League",
    "es.1.json": "La Liga",
    "de.1.json": "Bundesliga",
    "it.1.json": "Serie A",
    "fr.1.json": "Ligue 1",
}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_source(url):
    """Download public upstream data without disabling SSL verification."""
    request = Request(url, headers={"User-Agent": "PitchIQ-Educational-Match-Explorer/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=30, context=context) as response:
        return response.read().decode("utf-8-sig")


def match_row(date, season, competition, round_name, home, away, hg, ag, kickoff=""):
    hg, ag = int(hg), int(ag)
    return {
        "Date": date, "Season": season, "Competition": competition,
        "Round": round_name, "HomeTeam": home, "AwayTeam": away,
        "HomeGoals": hg, "AwayGoals": ag,
        "Result": "H" if hg > ag else ("A" if ag > hg else "D"),
        "Kickoff": kickoff,
    }


def parse_league(raw, season, competition):
    data = json.loads(raw)
    rows = []
    for match in data.get("matches", []):
        score = match.get("score") or {}
        # OpenFootball stores final scores as either [home, away] or {"ft": [...]}
        ft = score if isinstance(score, list) else score.get("ft")
        if not isinstance(ft, list) or len(ft) != 2 or any(x is None for x in ft):
            continue
        if pd.to_datetime(match["date"], errors="coerce") > pd.Timestamp.today().normalize():
            continue  # Never present a future-dated fixture as a completed result.
        rows.append(match_row(
            match["date"], season, competition, match.get("round", ""),
            match["team1"], match["team2"], ft[0], ft[1], match.get("time", ""),
        ))
    return rows


def parse_ucl(raw, season):
    """Parse OpenFootball Champions League cl.txt, mirroring research loader."""
    year = int(season[:4])
    date_re = re.compile(
        r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Za-z]{3})\s+"
        r"(\d{1,2})(?:\s+(\d{4}))?\s*$"
    )
    match_re = re.compile(
        r"^\s*(?:(\d{1,2}:\d{2})\s+)?(.+?\([A-Z]{3}\))\s+v\s+"
        r"(.+?\([A-Z]{3}\))\s+(\d+)-(\d+)(?:\s|$)"
    )
    rows, current_date, round_name, kickoff = [], None, "", ""
    for line in raw.splitlines():
        entry = line.strip()
        if entry.startswith("▪"):
            round_name = entry.lstrip("▪").strip()
            continue
        date_match = date_re.match(line)
        if date_match:
            month_text, day, explicit_year = date_match.groups()
            month = datetime.strptime(month_text, "%b").month
            match_year = int(explicit_year) if explicit_year else year if month >= 7 else year + 1
            current_date = f"{match_year}-{month:02d}-{int(day):02d}"
            kickoff = ""
            continue
        found = match_re.match(line)
        if found and current_date:
            time_text, home, away, hg, ag = found.groups()
            if time_text:
                kickoff = time_text
            rows.append(match_row(
                current_date, season, "Champions League", round_name,
                home.strip(), away.strip(), hg, ag, kickoff,
            ))
    return rows


@st.cache_data(ttl=86400, show_spinner="Loading real OpenFootball results…")
def historical_season(season):
    rows = []
    base = "https://raw.githubusercontent.com"
    for filename, competition in LEAGUES.items():
        url = f"{base}/openfootball/football.json/master/{season}/{filename}"
        rows.extend(parse_league(fetch_source(url), season, competition))
    if season != "2026-27":  # UCL 2026–27 source/format has not yet been verified.
        ucl_url = f"{base}/openfootball/champions-league/master/{season}/cl.txt"
        rows.extend(parse_ucl(fetch_source(ucl_url), season))
    if not rows:
        raise ValueError("OpenFootball returned no completed match records.")
    frame = validate_matches(pd.DataFrame(rows))
    if not set(LEAGUES.values()).issubset(set(frame["Competition"])):
        raise ValueError("At least one domestic competition has no valid matches.")
    return frame


with st.container(border=True):
    source = st.radio(
        "Match data source",
        ["Real OpenFootball results", "Fictional demo"],
        horizontal=True, index=1 if os.getenv("PITCHIQ_DATA_MODE") == "demo" else 0, key="match_explorer_source",
    )
    if source == "Real OpenFootball results":
        selected_season = st.selectbox("Historical season", SEASONS, index=1, key="real_explorer_season")
        st.caption("Completed results only · OpenFootball CC0. 2026–27 is in progress; UCL results currently run through 2025–26. Source updates may lag behind real matches.")
    else:
        selected_season = None
        st.caption("Fictional match results used to demonstrate the interface")

try:
    if source == "Real OpenFootball results":
        matches = historical_season(selected_season)
    else:
        # Explicitly use fictional data even if the local research datasets exist.
        from pitchiq.demo import matches as demo_matches
        matches = validate_matches(demo_matches())
except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
    st.error("Could not load match results. Check your internet connection or switch to Fictional demo.")
    with st.expander("Error details"):
        st.code(str(error))
    st.stop()

if matches.empty:
    st.info("There are no completed matches in this dataset.")
    st.stop()

with st.container(border=True):
    st.subheader("Find your matches")
    if source == "Real OpenFootball results":
        # The real season was already selected above; do not ask a second time.
        season = selected_season
        competition_col, team_col = st.columns(2)
    else:
        season_col, competition_col, team_col = st.columns(3)
        with season_col:
            seasons = sorted(matches["Season"].unique(), reverse=True)
            season_options = ["All seasons", *seasons]
            season = st.selectbox("Season", season_options, index=1, key="explorer_season_v2")
    filtered = matches if season == "All seasons" else matches[matches["Season"].eq(season)]
    with competition_col:
        options = ["All competitions", *sorted(filtered["Competition"].unique())]
        if st.session_state.get("explorer_competition_v2") not in options:
            st.session_state["explorer_competition_v2"] = options[0]
        competition = st.selectbox("Competition", options, key="explorer_competition_v2")
    if competition != "All competitions":
        filtered = filtered[filtered["Competition"].eq(competition)]
    with team_col:
        teams = sorted(set(filtered["HomeTeam"]) | set(filtered["AwayTeam"]))
        options = ["All teams", *teams]
        if st.session_state.get("explorer_team_v2") not in options:
            st.session_state["explorer_team_v2"] = options[0]
        team = st.selectbox("Team", options, key="explorer_team_v2")
    if team != "All teams":
        filtered = filtered[filtered["HomeTeam"].eq(team) | filtered["AwayTeam"].eq(team)]

    # Reset dates when the user changes competition, team or source.
    date_min = filtered["Date"].min().date() if not filtered.empty else matches["Date"].min().date()
    date_max = filtered["Date"].max().date() if not filtered.empty else matches["Date"].max().date()
    date_key = f"explorer_dates_{source}_{selected_season}_{season}_{competition}_{team}"
    date_range = st.date_input(
        "Date range", value=(date_min, date_max),
        min_value=matches["Date"].min().date(), max_value=matches["Date"].max().date(),
        key=date_key,
    )
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("Select an end date to complete the range.")
        st.stop()
    start, end = map(pd.Timestamp, date_range)
    filtered = filtered[filtered["Date"].between(start, end)]

if filtered.empty:
    st.info("No matches match the filters. Choose a different team or expand the dates.")
    st.stop()

if team == "All teams":
    labels = ["Home wins", "Draws", "Away wins"]
    values = [int(filtered["Result"].eq(result).sum()) for result in ("H", "D", "A")]
else:
    home = filtered["HomeTeam"].eq(team)
    away = filtered["AwayTeam"].eq(team)
    labels = ["Wins", "Draws", "Losses"]
    values = [
        int(((home & filtered["Result"].eq("H")) | (away & filtered["Result"].eq("A"))).sum()),
        int(filtered["Result"].eq("D").sum()),
        int(((home & filtered["Result"].eq("A")) | (away & filtered["Result"].eq("H"))).sum()),
    ]
for column, label, value in zip(st.columns(4), ["Matches", *labels], [len(filtered), *values]):
    column.metric(label, f"{value:,}")

chart_col, summary_col = st.columns([2, 1])
with chart_col:
    with st.container(border=True):
        st.subheader("Results at a glance")
        outcomes = pd.DataFrame({"Outcome": labels, "Matches": values})
        chart = alt.Chart(outcomes).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X("Matches:Q", title="Matches", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Outcome:N", title=None, sort=labels),
            color=alt.Color("Outcome:N", legend=None, scale=alt.Scale(domain=labels, range=PALETTE[:3])),
            tooltip=["Outcome:N", "Matches:Q"],
        ).properties(height=210)
        st.altair_chart(chart, width="stretch")
with summary_col:
    with st.container(border=True):
        st.subheader("Goal report")
        if team == "All teams":
            total_goals = int((filtered["HomeGoals"] + filtered["AwayGoals"]).sum())
            st.metric("Total goals", f"{total_goals:,}")
            st.metric("Goals per match", f"{total_goals / len(filtered):.2f}")
        else:
            scored = int(filtered["HomeGoals"].where(home, filtered["AwayGoals"]).sum())
            conceded = int(filtered["AwayGoals"].where(home, filtered["HomeGoals"]).sum())
            st.metric("Goals scored", f"{scored:,}", delta=f"{scored - conceded:+d} goal difference", delta_color="off")
            st.metric("Goals conceded", f"{conceded:,}")

st.subheader("Match results")
display = filtered.sort_values("Date", ascending=False).copy()
display["Score"] = display["HomeGoals"].astype(str) + " – " + display["AwayGoals"].astype(str)
columns = ["Date", "Season", "Competition", "HomeTeam", "Score", "AwayTeam", "Round"]
columns = [column for column in columns if column in display]
st.dataframe(display[columns], hide_index=True, width="stretch", column_config={
    "Date": st.column_config.DateColumn(format="DD MMM YYYY"),
    "HomeTeam": "Home team", "AwayTeam": "Away team",
})
st.caption(f"{len(filtered):,} matches · {start.date()} to {end.date()}")
download_csv(filtered, "pitchiq_match_results.csv", "Download filtered matches")
footer("Historical source: OpenFootball (CC0). Fictional mode: generated PitchIQ demo data.")
