
import json
import re
import ssl
import certifi
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import pandas as pd


# Settings
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "matches"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "all_matches.csv"

SEASONS = [
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24"
]

LEAGUES = {
    "en.1.json": "Premier League",
    "es.1.json": "La Liga",
    "de.1.json": "Bundesliga",
    "it.1.json": "Serie A",
    "fr.1.json": "Ligue 1"
}

LEAGUE_URL = (
    "https://raw.githubusercontent.com/"
    "openfootball/football.json/master"
)

UCL_URL = (
    "https://raw.githubusercontent.com/"
    "openfootball/champions-league/master"
)

EXPECTED_UCL_MATCHES = {
    "2020-21": 125,
    "2021-22": 125,
    "2022-23": 125,
    "2023-24": 125,
    "2024-25": 189
}


# Download missing files
def download_file(url, destination):
    if destination.exists():
        print(f"Using existing: {destination.name}")
        return

    print(f"Downloading: {destination.name}")

    request = Request(
        url,
        headers={
            "User-Agent": "PitchIQ-Student-Project"
        }
    )

    try:
       with urlopen(
    request,
    timeout=30,
    context=ssl.create_default_context(cafile=certifi.where())
    ) as response:
            content = response.read()

    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Could not download {url}: {error}"
        ) from error

    # Write only after the download succeeds
    destination.write_bytes(content)


def prepare_files():
    for season in SEASONS:
        season_dir = DATA_DIR / season
        season_dir.mkdir(exist_ok=True)

        for filename in LEAGUES:
            destination = season_dir / filename

            # Reuse existing 2022-23 downloads
            old_file = DATA_DIR / filename

            if (
                season == "2022-23"
                and not destination.exists()
                and old_file.exists()
            ):
                destination.write_bytes(
                    old_file.read_bytes()
                )

            url = (
                f"{LEAGUE_URL}/{season}/{filename}"
            )

            download_file(url, destination)

        ucl_destination = season_dir / "cl.txt"
        old_ucl = DATA_DIR / "cl.txt"

        if (
            season == "2022-23"
            and not ucl_destination.exists()
            and old_ucl.exists()
        ):
            ucl_destination.write_bytes(
                old_ucl.read_bytes()
            )

        download_file(
            f"{UCL_URL}/{season}/cl.txt",
            ucl_destination
        )


# Match result
def get_result(home_goals, away_goals):
    if home_goals > away_goals:
        return "H"

    if away_goals > home_goals:
        return "A"

    return "D"


def make_row(
    date,
    season,
    competition,
    round_name,
    home_team,
    away_team,
    home_goals,
    away_goals,
    kickoff=""
):
    return {
        "Date": date,
        "Season": season,
        "Competition": competition,
        "Round": round_name,
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "HomeGoals": home_goals,
        "AwayGoals": away_goals,
        "Result": get_result(
            home_goals,
            away_goals
        ),
        "Kickoff": kickoff
    }


# Load domestic league
def load_league(season, filename, competition):
    path = DATA_DIR / season / filename

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as file:
        data = json.load(file)

    rows = []

    for match in data["matches"]:
        score = match.get("score", {})
        full_time = score.get("ft")

        if (
            not isinstance(full_time, list)
            or len(full_time) != 2
            or any(value is None for value in full_time)
        ):
            continue

        home_goals = int(full_time[0])
        away_goals = int(full_time[1])

        rows.append(
            make_row(
                date=match["date"],
                season=season,
                competition=competition,
                round_name=match.get("round", ""),
                home_team=match["team1"],
                away_team=match["team2"],
                home_goals=home_goals,
                away_goals=away_goals,
                kickoff=match.get("time", "")
            )
        )

    return rows


# Load champions league
def load_ucl(season):
    path = DATA_DIR / season / "cl.txt"

    text = path.read_text(
        encoding="utf-8-sig"
    )

    start_year = int(season[:4])
    end_year = start_year + 1

    rows = []

    current_date = None
    current_round = ""
    current_time = ""

    date_pattern = re.compile(
        r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
        r"\s+([A-Za-z]{3})\s+(\d{1,2})"
        r"(?:\s+(\d{4}))?\s*$"
    )

    match_pattern = re.compile(
        r"^\s*(?:(\d{1,2}:\d{2})\s+)?"
        r"(.+?\([A-Z]{3}\))\s+v\s+"
        r"(.+?\([A-Z]{3}\))\s+"
        r"(\d+)-(\d+)(?:\s|$)"
    )

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("▪"):
            current_round = (
                stripped.lstrip("▪").strip()
            )
            continue

        date_match = date_pattern.match(line)

        if date_match:
            month_text, day, year = date_match.groups()

            month = datetime.strptime(
                month_text,
                "%b"
            ).month

            if year is None:
                year = (
                    start_year
                    if month >= 7
                    else end_year
                )

            current_date = (
                f"{int(year)}-{month:02d}-{int(day):02d}"
            )

            current_time = ""
            continue

        match = match_pattern.match(line)

        if match is None or current_date is None:
            continue

        time, home, away, hg, ag = match.groups()

        if time:
            current_time = time

        rows.append(
            make_row(
                date=current_date,
                season=season,
                competition="Champions League",
                round_name=current_round,
                home_team=home.strip(),
                away_team=away.strip(),
                home_goals=int(hg),
                away_goals=int(ag),
                kickoff=current_time
            )
        )

    return rows


# Load all seasons
def main():
    print("PitchIQ: Multi-Season Match Loader")
    print("---------------------------------")

    prepare_files()

    all_matches = []
    errors = []

    print("\nLoading competitions:\n")

    for season in SEASONS:
        print(f"Season: {season}")

        for filename, competition in LEAGUES.items():
            rows = load_league(
                season,
                filename,
                competition
            )

            if competition == "Bundesliga":
                expected = 306
            elif competition == "Ligue 1" and season >= "2023-24":
                expected = 306
            else:
                expected = 380
                
            print(
                f"  {competition}: {len(rows)}"
            )

            if len(rows) != expected:
                errors.append(
                    f"{season} {competition}: "
                    f"expected {expected}, got {len(rows)}"
                )

            all_matches.extend(rows)

        ucl_rows = load_ucl(season)

        print(
            f"  Champions League: {len(ucl_rows)}"
        )

        expected_ucl = EXPECTED_UCL_MATCHES[season]

        if len(ucl_rows) != expected_ucl:
            errors.append(
                f"{season} Champions League: "
                f"expected {expected_ucl}, "
                f"got {len(ucl_rows)}"
            )

        all_matches.extend(ucl_rows)
        print()

    df = pd.DataFrame(all_matches)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="raise"
    )

    duplicates = df.duplicated(
        subset=[
            "Season",
            "Date",
            "Competition",
            "HomeTeam",
            "AwayTeam"
        ]
    )

    if duplicates.any():
        errors.append(
            f"{duplicates.sum()} duplicate matches"
        )

    if errors:
        print("VALIDATION FAILED:")
        for error in errors:
            print(f"  - {error}")

        print(
            "\nExisting all_matches.csv "
            "has NOT been overwritten."
        )
        return

    df = df.sort_values(
        ["Date", "Competition", "HomeTeam"]
    ).reset_index(drop=True)

    df["Date"] = df["Date"].dt.strftime(
        "%Y-%m-%d"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("---------------------------------")
    print("SUCCESS!")
    print("---------------------------------")

    print(f"Total matches: {len(df)}")
    print(f"Seasons: {df['Season'].nunique()}")
    print(
        f"Competitions: "
        f"{df['Competition'].nunique()}"
    )

    print(f"\nSaved to: {OUTPUT_FILE}")

    print("\nMatches per season:")

    print(
        df.groupby("Season")
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
