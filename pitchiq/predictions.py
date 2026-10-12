"""Historical match features and independent Poisson scoreline estimates."""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson


def rolling_form(games, recent_count, side=None):
    if side is not None:
        games = [game for game in games if game["side"] == side]
    games = games[-recent_count:]
    if not games:
        return [np.nan] * 4
    count = len(games)
    return [
        sum(game["points"] for game in games) / count,
        sum(game["gf"] for game in games) / count,
        sum(game["ga"] for game in games) / count,
        sum(game["points"] == 3 for game in games) / count,
    ]


def history_snapshots(df, competition, season, recent_count, elo_k, elo_home_advantage):
    """Replay all match results up to the selected season's final recorded date.

    Results update in the same deterministic order as train_model_v2.py.
    Elo/form are tracked independently for each competition.
    """
    group = df[(df["Competition"] == competition) & (df["Season"] <= season)].copy()
    group = group.sort_values(["Date", "HomeTeam", "AwayTeam"])
    history = defaultdict(list)
    elo = defaultdict(lambda: 1500.0)
    last_played = {}
    last_date = None
    for date, day in group.groupby("Date", sort=True):
        last_date = date
        for match in day.itertuples(index=False):
            h, a = str(match.HomeTeam), str(match.AwayTeam)
            hg, ag = int(match.HomeGoals), int(match.AwayGoals)
            hp = 3 if hg > ag else (1 if hg == ag else 0)
            ap = 3 if ag > hg else (1 if ag == hg else 0)
            eh, ea = elo[h], elo[a]
            expected_h = 1 / (1 + 10 ** ((ea - (eh + elo_home_advantage)) / 400))
            actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            delta = elo_k * (actual_h - expected_h)
            elo[h], elo[a] = eh + delta, ea - delta
            history[h].append({"points": hp, "gf": hg, "ga": ag, "side": "H"})
            history[a].append({"points": ap, "gf": ag, "ga": hg, "side": "A"})
            last_played[h] = date
            last_played[a] = date
    return dict(history), dict(elo), dict(last_played), last_date


def make_feature_row(home, away, history, elo, last_played, as_of, recent_count, feature_names):
    hh, ah = history.get(home, []), history.get(away, [])
    parts = (
        rolling_form(hh, recent_count)
        + rolling_form(ah, recent_count)
        + rolling_form(hh, recent_count, "H")
        + rolling_form(ah, recent_count, "A")
        + [
            elo.get(home, 1500.0) - elo.get(away, 1500.0),
            elo.get(home, 1500.0),
            elo.get(away, 1500.0),
            (as_of - last_played[home]).days,
            (as_of - last_played[away]).days,
        ]
    )
    columns = (
        [f"HomeRecent_{v}" for v in ("Points", "Scored", "Conceded", "WinRate")]
        + [f"AwayRecent_{v}" for v in ("Points", "Scored", "Conceded", "WinRate")]
        + [f"HomeAtHome_{v}" for v in ("Points", "Scored", "Conceded", "WinRate")]
        + [f"AwayAtAway_{v}" for v in ("Points", "Scored", "Conceded", "WinRate")]
        + ["EloDifference", "HomeElo", "AwayElo", "HomeDaysRest", "AwayDaysRest"]
    )
    values = dict(zip(columns, parts))
    if set(feature_names) != set(values) or len(feature_names) != len(columns):
        raise ValueError("Saved V2 model feature schema does not match this page.")
    return pd.DataFrame([{column: values[column] for column in feature_names}])


def probabilities_from_model(model, X):
    probabilities = model.predict_proba(X)[0]
    return dict(zip(model.classes_, probabilities))


def scoreline_distribution(home_xg, away_xg, max_goals=12):
    """Independent Poisson scoreline approximation, normalized for tail truncation."""
    scores = np.arange(max_goals + 1)
    home_p = poisson.pmf(scores, home_xg)
    away_p = poisson.pmf(scores, away_xg)
    grid = np.outer(home_p, away_p)
    grid /= grid.sum()
    positions = np.dstack(np.unravel_index(np.argsort(grid.ravel())[::-1], grid.shape))[0]
    return [(int(h), int(a), float(grid[h, a])) for h, a in positions[:5]]


