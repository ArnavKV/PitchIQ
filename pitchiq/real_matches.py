"""Read OpenFootball results and keep unplayed fixtures separate.

Current-season score placeholders are not treated as completed matches.
"""
import json
import re
import ssl
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import certifi
import pandas as pd
import streamlit as st

from pitchiq.data import validate_matches

SEASONS = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2020, 2027))
CURRENT_SEASON = "2026-27"
LEAGUES = {
    "en.1.json": "Premier League", "es.1.json": "La Liga",
    "de.1.json": "Bundesliga", "it.1.json": "Serie A", "fr.1.json": "Ligue 1",
}
BASE = "https://raw.githubusercontent.com/openfootball"
FIXTURE_COLUMNS = ["Date", "Season", "Competition", "Round", "HomeTeam", "AwayTeam", "Kickoff"]


def today_utc():
    return pd.Timestamp(datetime.now(timezone.utc).date())


@st.cache_data(ttl=3600, show_spinner=False)
def _get(url):
    req = Request(url, headers={"User-Agent": "PitchIQ-Public-Educational/1.1"})
    with urlopen(req, timeout=25, context=ssl.create_default_context(cafile=certifi.where())) as response:
        return response.read().decode("utf-8-sig")


def _completed_row(m, season, competition, ft):
    hg, ag = (int(x) for x in ft)
    if hg < 0 or ag < 0:
        return None
    return {
        "Date": m["date"], "Season": season, "Competition": competition,
        "Round": m.get("round", ""), "HomeTeam": m["team1"],
        "AwayTeam": m["team2"], "HomeGoals": hg, "AwayGoals": ag,
        "Result": "H" if hg > ag else "A" if ag > hg else "D",
        "Kickoff": m.get("time", ""),
    }


def parse_league(raw, season, competition, as_of=None):
    """Return (completed results, remaining/unverified fixtures).

    ``as_of`` represents today's date, not a historical replay cutoff.
    An ``ft`` final score is trusted only on or before as_of. Older finished
    seasons may contain legacy list-style scores. List-style scores in the
    active season are ambiguous; do not interpret placeholders as draws.
    """
    if as_of is None:
        as_of = today_utc()
    as_of = pd.Timestamp(as_of).normalize()
    data = json.loads(raw)
    completed, fixtures = [], []
    for m in data.get("matches", []):
        if not all(m.get(k) for k in ("date", "team1", "team2")):
            continue
        date = pd.to_datetime(m["date"], errors="coerce")
        if pd.isna(date):
            continue
        score = m.get("score")
        ft = score.get("ft") if isinstance(score, dict) else None
        # Some older files store the final score as a two-number list.
        if ft is None and isinstance(score, list) and season != CURRENT_SEASON:
            ft = score
        valid = (
            isinstance(ft, list) and len(ft) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in ft)
        )
        if valid and date <= as_of:
            record = _completed_row(m, season, competition, ft)
            if record:
                completed.append(record)
            continue
        fixtures.append({
            "Date": m["date"], "Season": season, "Competition": competition,
            "Round": m.get("round", ""), "HomeTeam": m["team1"],
            "AwayTeam": m["team2"], "Kickoff": m.get("time", ""),
        })
    return completed, fixtures


def parse_ucl(raw, season, as_of=None):
    """Get completed OpenFootball text-format UCL results (no bracket inference)."""
    if as_of is None:
        as_of = today_utc()
    year = int(season[:4])
    date_re = re.compile(r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Za-z]{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
    match_re = re.compile(r"^\s*(?:(\d{1,2}:\d{2})\s+)?(.+?\([A-Z]{3}\))\s+v\s+(.+?\([A-Z]{3}\))\s+(\d+)-(\d+)(?:\s|$)")
    rows, date, round_, kickoff = [], None, "", ""
    for line in raw.splitlines():
        x = line.strip()
        if x.startswith("▪"):
            round_ = x.lstrip("▪").strip()
            continue
        m = date_re.match(line)
        if m:
            month, day, specified = m.groups()
            month = datetime.strptime(month, "%b").month
            actual_year = int(specified) if specified else (year if month >= 7 else year + 1)
            date = f"{actual_year:04d}-{month:02d}-{int(day):02d}"
            kickoff = ""
            continue
        m = match_re.match(line)
        if m and date and pd.Timestamp(date) <= pd.Timestamp(as_of):
            time, home, away, hg, ag = m.groups()
            if time:
                kickoff = time
            row = _completed_row({"date": date, "round": round_, "team1": home.strip(), "team2": away.strip(), "time": kickoff}, season, "Champions League", [int(hg), int(ag)])
            rows.append(row)
    return rows


@st.cache_data(ttl=3600, show_spinner="Loading OpenFootball season…")
def load_real_season(season):
    if season not in SEASONS:
        raise ValueError(f"Unsupported OpenFootball season: {season}")
    rows, planned = [], []
    for filename, competition in LEAGUES.items():
        raw = _get(f"{BASE}/football.json/master/{season}/{filename}")
        completed, fixtures = parse_league(raw, season, competition)
        rows.extend(completed)
        planned.extend(fixtures)
    if season != CURRENT_SEASON:
        ucl_raw = _get(f"{BASE}/champions-league/master/{season}/cl.txt")
        rows.extend(parse_ucl(ucl_raw, season))
    if not rows:
        raise ValueError(f"No completed results for {season}; source may not be updated")
    matches = validate_matches(pd.DataFrame(rows))
    fixtures = pd.DataFrame(planned, columns=FIXTURE_COLUMNS)
    if not fixtures.empty:
        fixtures["Date"] = pd.to_datetime(fixtures["Date"], errors="coerce")
        fixtures = fixtures.dropna(subset=["Date"])
        fixtures = fixtures.drop_duplicates(["Date", "Competition", "HomeTeam", "AwayTeam"])
        fixtures = fixtures.sort_values(["Date", "Competition", "HomeTeam"]).reset_index(drop=True)
    return matches, fixtures


@st.cache_data(ttl=3600, show_spinner="Loading real match history for modeling…")
def load_real_matches(upto=CURRENT_SEASON):
    if upto not in SEASONS:
        raise ValueError(f"Unsupported season: {upto}")
    frames = [load_real_season(season)[0] for season in SEASONS if season <= upto]
    return validate_matches(pd.concat(frames, ignore_index=True))


def load_real_fixtures(season=CURRENT_SEASON):
    return load_real_season(season)[1].copy()
