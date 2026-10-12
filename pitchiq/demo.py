"""Generated example data and models for the public demo.

Everything here is generated from fixed random seeds. No football provider data,
downloaded model artifact, real player identity, or real match result is read.
The models demonstrate the application workflow; their generated holdout scores
do not establish accuracy on real football matches.
"""
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pitchiq.shots import shot_geometry
from pitchiq.simulation import HOME_ADV, K, RECENT, feature_row

SEASONS = ("2020-21", "2021-22", "2022-23", "2023-24")
DOMESTIC_COMPETITION = "Demo Premier League"
CUP_COMPETITION = "Demo Champions League"
DATA_SOURCE = "PitchIQ original synthetic demo"
MATCH_FEATURES = tuple(
    f"{prefix}_{metric}"
    for prefix in ("HomeRecent", "AwayRecent", "HomeAtHome", "AwayAtAway")
    for metric in ("Points", "Scored", "Conceded", "WinRate")
) + ("EloDifference", "HomeElo", "AwayElo", "HomeDaysRest", "AwayDaysRest")
SHOT_FEATURES = ("ShotDistance", "ShotAngle", "BodyPart", "ShotType")
DOMESTIC_TEAMS = tuple(
    f"Demo {name}"
    for name in (
        "Amberfen United", "Brindlewick Rovers", "Coppervale FC", "Dappleford Athletic",
        "Emberbrook City", "Fernspire Wanderers", "Glimmerford Town", "Hazelcrest FC",
        "Ivystone United", "Juniperwick City", "Kestrelfen Rovers", "Lanternvale Athletic",
        "Meadowspire FC", "Northglade Town", "Oakwhistle United", "Pebblecrest City",
        "Quartzfen Rovers", "Reedspire Athletic", "Silverglade FC", "Thistlewick Town",
    )
)
CUP_TEAMS = tuple(f"Demo Cup Club {number:02}" for number in range(1, 17))
PLAYERS = tuple(f"Demo Striker {number:02}" for number in range(1, 13))
KEEPERS = tuple(f"Demo Keeper {number:02}" for number in range(1, 5))


def _round_robin(teams):
    """Each team plays once per round, with every home/away pairing once."""
    rotation = list(teams)
    first_half = []
    for round_number in range(len(teams) - 1):
        pairs = []
        for index in range(len(teams) // 2):
            first, second = rotation[index], rotation[-index - 1]
            pairs.append((first, second) if (round_number + index) % 2 else (second, first))
        first_half.append(pairs)
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return first_half + [[(away, home) for home, away in pairs] for pairs in first_half]


@lru_cache(maxsize=1)
def _matches():
    rng = np.random.default_rng(20261011)
    rows = []
    for competition, teams, interval in (
        (DOMESTIC_COMPETITION, DOMESTIC_TEAMS, 7),
        (CUP_COMPETITION, CUP_TEAMS, 9),
    ):
        strengths = dict(zip(teams, rng.normal(0, 0.25, len(teams))))
        for season in SEASONS:
            start = pd.Timestamp(f"{season[:4]}-08-19")
            if competition == CUP_COMPETITION:
                start += pd.Timedelta(days=21)
            for round_number, pairs in enumerate(_round_robin(teams)):
                date = start + pd.Timedelta(days=round_number * interval)
                for home, away in pairs:
                    difference = strengths[home] - strengths[away]
                    home_goals = int(rng.poisson(np.exp(0.35 + difference)))
                    away_goals = int(rng.poisson(np.exp(0.12 - difference)))
                    rows.append({
                        "Date": date, "Season": season, "Competition": competition,
                        "HomeTeam": home, "AwayTeam": away,
                        "HomeGoals": home_goals, "AwayGoals": away_goals,
                        "Result": "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D",
                        "Synthetic": True, "Source": DATA_SOURCE,
                    })
    return pd.DataFrame(rows).sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def matches():
    """Return a fresh frame of four entirely fictional completed seasons."""
    return _matches().copy(deep=True)


@lru_cache(maxsize=1)
def _shots():
    rng = np.random.default_rng(20261012)
    rows = []
    for index in range(1200):
        player_index = index % len(PLAYERS)
        keeper_index = int(rng.integers(len(KEEPERS)))
        shot_type = str(rng.choice(["Open Play", "Free Kick", "Penalty"], p=[0.8, 0.13, 0.07]))
        body_part = str(rng.choice(["Right Foot", "Left Foot", "Head"], p=[0.55, 0.3, 0.15]))
        if shot_type != "Open Play" and body_part == "Head":
            body_part = "Right Foot"
        x, y = (108.0, 40.0) if shot_type == "Penalty" else (float(rng.uniform(82, 118)), float(rng.uniform(12, 68)))
        distance, angle = shot_geometry(x, y)
        # This equation is invented for the demo, not calibrated to provider data.
        log_odds = (
            -1.4 - 0.065 * distance + 0.034 * angle
            - 0.45 * (body_part == "Head") + 1.7 * (shot_type == "Penalty")
            - 0.3 * (shot_type == "Free Kick") + (player_index - 5.5) * 0.035
            - (keeper_index - 1.5) * 0.08 * (shot_type != "Penalty")
        )
        probability = float(1 / (1 + np.exp(-log_odds)))
        rows.append({
            "ShotID": f"demo-shot-{index + 1:04}", "MatchID": f"demo-shot-match-{index // 10 + 1:03}",
            "Shooter": PLAYERS[player_index], "ShooterID": 9001 + player_index,
            "Goalkeeper": KEEPERS[keeper_index], "GoalkeeperID": 9101 + keeper_index,
            "X": x, "Y": y, "ShotDistance": distance, "ShotAngle": angle,
            "BodyPart": body_part, "ShotType": shot_type,
            "Goal": int(rng.random() < probability),
            # Legacy UI schema only: this value is synthetic and is NOT StatsBomb xG.
            "StatsBomb_xG": probability,
            "Synthetic": True, "Source": DATA_SOURCE,
        })
    return pd.DataFrame(rows)


def shots():
    """Return invented chances; StatsBomb_xG is a compatibility column only."""
    return _shots().copy(deep=True)


@lru_cache(maxsize=1)
def _player_frames():
    rng = np.random.default_rng(20261013)
    core_rows, advanced_rows = [], []
    for year in range(2020, 2024):
        for index, player in enumerate(PLAYERS):
            games = int(rng.integers(20, 39))
            minutes = games * int(rng.integers(55, 88))
            goals = int(rng.integers(5, 26))
            assists = int(rng.integers(1, 13))
            common = {
                "Player": player, "ProviderPlayerID": f"demo:{9001 + index}",
                "Games": games, "Goals": goals, "Assists": assists, "Minutes": minutes,
                "Synthetic": True, "Source": DATA_SOURCE,
            }
            core_rows.append({**common, "season": year, "competition_id": DOMESTIC_COMPETITION})
            advanced_rows.append({
                **common, "Season": year, "league": DOMESTIC_COMPETITION,
                "Shots": int(rng.integers(max(40, goals), 111)),
                "xG": float(np.round(goals * rng.uniform(0.75, 1.25), 2)),
                "xA": float(np.round(assists * rng.uniform(0.75, 1.25), 2)),
            })
    return pd.DataFrame(core_rows), pd.DataFrame(advanced_rows)


def player_frames():
    """Return fictional core/advanced frames matching the two local CSV schemas.

    Identical demo IDs intentionally link the same fictional player across both
    frames. Neither frame contains Transfermarkt or Understat observations.
    """
    return tuple(frame.copy(deep=True) for frame in _player_frames())


def _match_features(frame):
    rows = []
    for _, competition in frame.groupby("Competition", sort=True):
        history = defaultdict(list)
        elo = defaultdict(lambda: 1500.0)
        last = {}
        for date, day in competition.groupby("Date", sort=True):
            updates = []
            for match in day.itertuples(index=False):
                home, away = match.HomeTeam, match.AwayTeam
                rows.append({
                    **dict(zip(MATCH_FEATURES, feature_row(home, away, date, history, elo, last, MATCH_FEATURES))),
                    "Result": match.Result, "HomeGoals": match.HomeGoals,
                    "AwayGoals": match.AwayGoals, "Season": match.Season,
                })
                expected = 1 / (1 + 10 ** ((elo[away] - (elo[home] + HOME_ADV)) / 400))
                actual = 1.0 if match.Result == "H" else 0.5 if match.Result == "D" else 0.0
                updates.append((match, K * (actual - expected)))
            # Features for every same-day match use only earlier match dates.
            for match, change in updates:
                home, away = match.HomeTeam, match.AwayTeam
                elo[home] += change
                elo[away] -= change
                home_points = 3 if match.Result == "H" else 1 if match.Result == "D" else 0
                away_points = 3 if match.Result == "A" else 1 if match.Result == "D" else 0
                history[home].append({"points": home_points, "gf": match.HomeGoals, "ga": match.AwayGoals, "side": "H"})
                history[away].append({"points": away_points, "gf": match.AwayGoals, "ga": match.HomeGoals, "side": "A"})
                last[home] = last[away] = date
    return pd.DataFrame(rows)


def _metadata():
    return {
        "synthetic": True, "data_source": "synthetic",
        "notes": "Illustrative model fitted only to original generated data. Synthetic holdout metrics do not measure real-world performance.",
    }


@lru_cache(maxsize=1)
def _match_models():
    frame = _match_features(_matches())
    train = frame[frame.Season != SEASONS[-1]]
    test = frame[frame.Season == SEASONS[-1]]
    X_train, X_test = train[list(MATCH_FEATURES)], test[list(MATCH_FEATURES)]
    outcome = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500, random_state=42))
    home = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))
    away = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))
    outcome.fit(X_train, train.Result)
    home.fit(X_train, train.HomeGoals)
    away.fit(X_train, train.AwayGoals)
    common = {
        **_metadata(), "features": list(MATCH_FEATURES), "recent_matches": RECENT,
        "minimum_history": 3, "elo_k_factor": K, "home_elo_advantage": HOME_ADV,
        "training_matches": len(train), "testing_matches": len(test),
        "test_season": SEASONS[-1],
    }
    return {
        "match_predictor_v2.joblib": {
            **common, "model": outcome,
            "accuracy": float(outcome.score(X_test, test.Result)),
            "log_loss": float(log_loss(test.Result, outcome.predict_proba(X_test), labels=outcome.classes_)),
        },
        "match_predictor_v3.joblib": {
            **common, "poisson_home_model": home, "poisson_away_model": away,
            "home_goals_mae": float(mean_absolute_error(test.HomeGoals, home.predict(X_test))),
            "away_goals_mae": float(mean_absolute_error(test.AwayGoals, away.predict(X_test))),
        },
    }


def _profile(group):
    reference = float(group.StatsBomb_xG.mean())
    goals = int(group.Goal.sum())
    observed = (goals + 20 * reference) / (len(group) + 20)
    multiplier = (observed / (1 - observed)) / (reference / (1 - reference))
    return {
        "shots": len(group), "goals": goals, "reference_xg": float(group.StatsBomb_xG.sum()),
        "multiplier": float(np.clip(multiplier, 0.65, 1.4)),
    }


@lru_cache(maxsize=1)
def _xg_model():
    frame = _shots()
    # Complete fictional matches, rather than individual shots, are held out.
    train_ids = frame.MatchID.drop_duplicates().iloc[:96]
    train = frame[frame.MatchID.isin(train_ids)]
    test = frame[~frame.MatchID.isin(train_ids)]
    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), ["ShotDistance", "ShotAngle"]),
        ("category", OneHotEncoder(handle_unknown="ignore"), ["BodyPart", "ShotType"]),
    ])
    classifier = make_pipeline(transformer, LogisticRegression(max_iter=500, random_state=42))
    classifier.fit(train[list(SHOT_FEATURES)], train.Goal)
    goals_index = list(classifier.classes_).index(1)
    probabilities = classifier.predict_proba(test[list(SHOT_FEATURES)])[:, goals_index]
    players = {str(int(pid)): _profile(group) for pid, group in train.groupby("ShooterID")}
    feet = {f"{int(pid)}|{part}": _profile(group) for (pid, part), group in train.groupby(["ShooterID", "BodyPart"])}
    keepers = {}
    for keeper, group in train[train.ShotType != "Penalty"].groupby("Goalkeeper"):
        profile = _profile(group)
        keepers[keeper] = {
            "shots": profile["shots"], "goals_conceded": profile["goals"],
            "reference_xg_against": profile["reference_xg"], "multiplier": profile["multiplier"],
        }
    return {
        **_metadata(), "model": classifier, "features": list(SHOT_FEATURES),
        "training_shots": len(train), "testing_shots": len(test),
        "training_matches": len(train_ids), "testing_matches": test.MatchID.nunique(),
        "brier_score": float(brier_score_loss(test.Goal, probabilities)),
        "player_factors": players, "player_foot_factors": feet, "goalkeeper_factors": keepers,
    }


def model(filename):
    """Build a trusted in-memory demo model with the legacy artifact interface.

    Filenames select an implementation; they are never opened or deserialized.
    Estimators are cached for efficient interactive reruns. Callers should treat
    the fitted estimators as read-only; all metadata is returned as a fresh copy.
    """
    if filename == "xg_predictor.joblib":
        artifact = _xg_model()
    elif filename in ("match_predictor_v2.joblib", "match_predictor_v3.joblib"):
        artifact = _match_models()[filename]
    else:
        raise ValueError(f"No synthetic demo model is available for {filename!r}.")
    return {
        key: value if key in ("model", "poisson_home_model", "poisson_away_model") else deepcopy(value)
        for key, value in artifact.items()
    }
