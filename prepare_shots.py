"""Build a multi-competition shot dataset from StatsBomb Open Data.

Run from PitchIQ repository root: python prepare_shots.py
Source: https://github.com/statsbomb/open-data
"""
import json
import ssl
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi
import pandas as pd

ROOT = Path(__file__).resolve().parent
SHOT_DIR = ROOT / "data" / "shots"
CACHE_DIR = SHOT_DIR / "raw_events"
MATCH_DIR = SHOT_DIR / "match_lists"
for folder in (SHOT_DIR, CACHE_DIR, MATCH_DIR):
    folder.mkdir(parents=True, exist_ok=True)

OUTPUT = SHOT_DIR / "shots.csv"
PARTIAL_OUTPUT = SHOT_DIR / "shots_partial.csv"
MANIFEST = SHOT_DIR / "coverage.csv"
BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Include Messi/Ronaldo-era seasons. Change to None to include older seasons too.
MIN_START_YEAR = 2008
# Only selected competitions; names match StatsBomb competitions.json exactly.
COMPETITIONS = {
    "La Liga", "Champions League", "Premier League", "1. Bundesliga",
    "Ligue 1", "Serie A", "FIFA World Cup", "UEFA Euro",
    "Copa America", "Major League Soccer",
}
# Include every available match. For a quick test, use 5 instead.
MATCH_LIMIT_PER_SEASON = None
# Set to e.g. 2 for a quick pilot. None means all discovered seasons.
SEASON_LIMIT = None

CONTEXT = ssl.create_default_context(cafile=certifi.where())


def fetch_json(url, cache_file):
    """Read existing JSON if possible; otherwise download and cache atomically."""
    if cache_file.exists():
        try:
            with cache_file.open(encoding="utf-8") as handle:
                return json.load(handle), True
        except (OSError, ValueError):
            print(f"  Invalid cache: {cache_file.name}; re-downloading")

    request = Request(url, headers={"User-Agent": "PitchIQ-Student-Analytics/1.0"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45, context=CONTEXT) as response:
                data = json.load(response)
            temp = cache_file.with_suffix(".tmp")
            with temp.open("w", encoding="utf-8") as handle:
                json.dump(data, handle)
            temp.replace(cache_file)
            return data, False
        except HTTPError as exc:
            if exc.code in (403, 404) or attempt == 2:
                raise
        except (URLError, TimeoutError, OSError, ValueError):
            if attempt == 2:
                raise
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}")


def season_year(name):
    """For 2023/2024, 2022, etc. use its starting year."""
    try:
        return int(str(name)[:4])
    except ValueError:
        return -1


def discover_seasons():
    data, cached = fetch_json(
        f"{BASE_URL}/competitions.json", SHOT_DIR / "competitions.json"
    )
    selected = [
        c for c in data
        if c.get("competition_name") in COMPETITIONS
        and c.get("competition_gender") == "male"
        and (MIN_START_YEAR is None or season_year(c.get("season_name")) >= MIN_START_YEAR)
    ]
    selected.sort(key=lambda c: (
        c["competition_name"], season_year(c["season_name"]), c["season_id"]
    ))
    if SEASON_LIMIT is not None:
        selected = selected[:SEASON_LIMIT]
    return selected


def shot_rows(events, match, competition, season):
    rows = []
    home = match.get("home_team", {}).get("home_team_name", "Unknown")
    away = match.get("away_team", {}).get("away_team_name", "Unknown")
    for event in events:
        if event.get("type", {}).get("name") != "Shot":
            continue
        loc = event.get("location") or []
        if len(loc) < 2 or not all(isinstance(v, (int, float)) for v in loc[:2]):
            continue
        x, y = map(float, loc[:2])
        if not (0 <= x <= 120 and 0 <= y <= 80):
            continue
        shot = event.get("shot") or {}
        shooter = event.get("player") or {}
        team = event.get("team") or {}
        team_name = team.get("name", "Unknown")
        outcome = (shot.get("outcome") or {}).get("name", "Unknown")
        # Goalkeeper identity is only reliable when a visible opposition GK
        # appears in the shot freeze frame; leave missing values blank.
        goalkeeper_names = [
            (p.get("player") or {}).get("name")
            for p in (shot.get("freeze_frame") or [])
            if p.get("teammate") is False
            and (p.get("position") or {}).get("name") == "Goalkeeper"
            and (p.get("player") or {}).get("name")
        ]
        rows.append({
            "EventID": event.get("id"),
            "MatchID": match["match_id"],
            "Date": match.get("match_date"),
            "Competition": competition,
            "Season": season,
            "Shooter": shooter.get("name", "Unknown"),
            "ShooterID": shooter.get("id"),
            "Team": team_name,
            "TeamID": team.get("id"),
            "Opponent": away if team_name == home else home if team_name == away else "Unknown",
            "HomeTeam": home,
            "AwayTeam": away,
            "Goalkeeper": goalkeeper_names[0] if len(goalkeeper_names) == 1 else None,
            "X": x,
            "Y": y,
            "BodyPart": (shot.get("body_part") or {}).get("name", "Unknown"),
            "ShotType": (shot.get("type") or {}).get("name", "Unknown"),
            "Outcome": outcome,
            "Goal": int(outcome == "Goal"),
            "StatsBomb_xG": shot.get("statsbomb_xg"),
        })
    return rows


def main():
    print("=" * 54)
    print("PITCHIQ | STATS BOMB HISTORICAL SHOT DATA")
    print("=" * 54)
    seasons = discover_seasons()
    print(f"Discovered {len(seasons)} competition-seasons")
    print("Source: StatsBomb Open Data")
    all_rows, coverage, failures = [], [], []
    seen_matches = set()
    reused = 0
    for i, entry in enumerate(seasons, 1):
        competition = entry["competition_name"]
        season = entry["season_name"]
        comp_id, season_id = entry["competition_id"], entry["season_id"]
        print(f"\n[{i}/{len(seasons)}] {competition} {season}", flush=True)
        try:
            matches, _ = fetch_json(
                f"{BASE_URL}/matches/{comp_id}/{season_id}.json",
                MATCH_DIR / f"{comp_id}_{season_id}.json",
            )
            if not isinstance(matches, list):
                raise ValueError("Match list isn't a list")
        except Exception as exc:
            print(f"  MATCH LIST FAILED: {exc}")
            failures.append(f"{competition} {season}: match list: {exc}")
            continue
        matches.sort(key=lambda m: (m.get("match_date", ""), m["match_id"]))
        if MATCH_LIMIT_PER_SEASON is not None:
            matches = matches[:MATCH_LIMIT_PER_SEASON]
        successful, season_shots = 0, 0
        for j, match in enumerate(matches, 1):
            match_id = match["match_id"]
            if match_id in seen_matches:
                continue
            try:
                events, cached = fetch_json(
                    f"{BASE_URL}/events/{match_id}.json",
                    CACHE_DIR / f"{match_id}.json",
                )
                if not isinstance(events, list):
                    raise ValueError("Events aren't a list")
                rows = shot_rows(events, match, competition, season)
                all_rows.extend(rows)
                seen_matches.add(match_id)
                successful += 1
                season_shots += len(rows)
                reused += int(cached)
                print(f"  {j}/{len(matches)} match {match_id}: {len(rows)} shots"
                      f"{' [cached]' if cached else ''}", flush=True)
            except Exception as exc:
                failures.append(f"{competition} {season} match {match_id}: {exc}")
                print(f"  {j}/{len(matches)} FAILED {match_id}: {exc}", flush=True)
        coverage.append({
            "Competition": competition, "Season": season,
            "ListedMatches": len(matches), "ProcessedMatches": successful,
            "Shots": season_shots,
        })

    if not all_rows:
        print("\nNo shots collected. Existing CSV was not modified.")
        return
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["MatchID", "EventID"])
    df = df.sort_values(["Date", "MatchID", "EventID"]).reset_index(drop=True)
    pd.DataFrame(coverage).to_csv(MANIFEST, index=False)
    path = PARTIAL_OUTPUT if failures else OUTPUT
    df.to_csv(path, index=False)
    print("\n" + "=" * 54)
    print("PARTIAL RESULT" if failures else "SUCCESS")
    print(f"Competitions: {df['Competition'].nunique()}")
    print(f"Matches: {df['MatchID'].nunique()}")
    print(f"Shots: {len(df)} | Goals: {int(df['Goal'].sum())}")
    print(f"Unique shooter IDs: {df['ShooterID'].nunique()}")
    print(f"Goalkeeper identified on {df['Goalkeeper'].notna().sum()} shots")
    print(f"Cached event files reused: {reused}")
    print(f"Dataset: {path}")
    print(f"Coverage report: {MANIFEST}")
    for name in ("Messi", "Ronaldo"):
        found = df[df["Shooter"].str.contains(name, case=False, na=False)]
        print(f"\nShooter names containing '{name}':")
        if found.empty:
            print("  No matches found in downloaded data")
        else:
            print(found.groupby("Shooter").agg(
                Shots=("Goal", "size"), Goals=("Goal", "sum")
            ).to_string())
    if failures:
        print(f"\n{len(failures)} failed downloads. Existing shots.csv kept intact.")
        for failure in failures[:20]:
            print(" -", failure)
        print("Re-run to retry; completed downloads are cached.")
    print("\nData credit: StatsBomb Open Data (logo required in published work).")


if __name__ == "__main__":
    main()
