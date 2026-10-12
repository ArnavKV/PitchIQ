"""Historical standings and reproducible Monte Carlo league/cup simulations."""
from collections import defaultdict

import numpy as np
import pandas as pd

RECENT = 5
K = 20.0
HOME_ADV = 60.0


def form(games, side=None):
    if side:
        games = [g for g in games if g["side"] == side]
    games = games[-RECENT:]
    if not games:
        return [np.nan] * 4
    n = len(games)
    return [sum(g["points"] for g in games) / n,
            sum(g["gf"] for g in games) / n,
            sum(g["ga"] for g in games) / n,
            sum(g["points"] == 3 for g in games) / n]


def snapshot(all_matches, competition, cutoff):
    """Same competition-isolated Elo/form conventions as train_model_v2.

    cutoff is exclusive: no match played on the cutoff date enters team form.
    Same-date results are processed together.
    """
    past = all_matches[(all_matches.Competition == competition) & (all_matches.Date < cutoff)]
    hist = defaultdict(list)
    elo = defaultdict(lambda: 1500.0)
    last = {}
    for date, day in past.groupby("Date", sort=True):
        updates = []
        for row in day.itertuples(index=False):
            h, a = str(row.HomeTeam), str(row.AwayTeam)
            hg, ag = int(row.HomeGoals), int(row.AwayGoals)
            hp = 3 if hg > ag else (1 if hg == ag else 0)
            ap = 3 if ag > hg else (1 if ag == hg else 0)
            expected = 1 / (1 + 10 ** ((elo[a] - (elo[h] + HOME_ADV)) / 400))
            actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            updates.append((h, a, hg, ag, hp, ap, K * (actual - expected)))
        for h, a, hg, ag, hp, ap, change in updates:
            elo[h] += change
            elo[a] -= change
            hist[h].append({"points": hp, "gf": hg, "ga": ag, "side": "H"})
            hist[a].append({"points": ap, "gf": ag, "ga": hg, "side": "A"})
            last[h] = date
            last[a] = date
    return dict(hist), dict(elo), dict(last)


def feature_row(home, away, date, hist, elo, last, columns):
    pieces = (form(hist.get(home, [])) + form(hist.get(away, []))
              + form(hist.get(home, []), "H") + form(hist.get(away, []), "A")
              + [elo.get(home, 1500.0) - elo.get(away, 1500.0),
                 elo.get(home, 1500.0), elo.get(away, 1500.0),
                 min(60, max(0, (date - last[home]).days)) if home in last else np.nan,
                 min(60, max(0, (date - last[away]).days)) if away in last else np.nan])
    names = ([f"HomeRecent_{x}" for x in ("Points", "Scored", "Conceded", "WinRate")]
             + [f"AwayRecent_{x}" for x in ("Points", "Scored", "Conceded", "WinRate")]
             + [f"HomeAtHome_{x}" for x in ("Points", "Scored", "Conceded", "WinRate")]
             + [f"AwayAtAway_{x}" for x in ("Points", "Scored", "Conceded", "WinRate")]
             + ["EloDifference", "HomeElo", "AwayElo", "HomeDaysRest", "AwayDaysRest"])
    values = dict(zip(names, pieces))
    if set(columns) != set(names):
        raise ValueError("Model's feature columns do not match simulator's 21 pre-match features")
    return [values[col] for col in columns]


def predict_many(pairs, hist, elo, last, model):
    if not pairs:
        return np.empty((0, 2))
    cols = list(model["features"])
    X = pd.DataFrame([feature_row(h, a, d, hist, elo, last, cols) for h, a, d in pairs], columns=cols)
    home = np.clip(model["poisson_home_model"].predict(X), 0.05, 8.0)
    away = np.clip(model["poisson_away_model"].predict(X), 0.05, 8.0)
    return np.stack([home, away], axis=1)


def current_table(played, teams):
    lookup = {team: i for i, team in enumerate(teams)}
    points = np.zeros(len(teams), dtype=int)
    gf = np.zeros(len(teams), dtype=int)
    ga = np.zeros(len(teams), dtype=int)
    wins = np.zeros(len(teams), dtype=int)
    draws = np.zeros(len(teams), dtype=int)
    losses = np.zeros(len(teams), dtype=int)
    for row in played.itertuples(index=False):
        h, a = lookup[row.HomeTeam], lookup[row.AwayTeam]
        hg, ag = int(row.HomeGoals), int(row.AwayGoals)
        gf[h] += hg; ga[h] += ag; gf[a] += ag; ga[a] += hg
        if hg > ag:
            points[h] += 3; wins[h] += 1; losses[a] += 1
        elif ag > hg:
            points[a] += 3; wins[a] += 1; losses[h] += 1
        else:
            points[h] += 1; points[a] += 1; draws[h] += 1; draws[a] += 1
    frame = pd.DataFrame({"Team": teams, "P": wins + draws + losses,
                          "W": wins, "D": draws, "L": losses,
                          "GF": gf, "GA": ga, "GD": gf - ga, "Pts": points})
    return frame.sort_values(["Pts", "GD", "GF", "Team"], ascending=[False, False, False, True]).reset_index(drop=True)


def league_simulate(played, remaining, teams, mus, count, seed):
    """Vectorized independent Poisson simulations, with fixed cutoff estimates.

    Ties sorted by points, goal difference, goals scored; real-world head-to-head
    tie-breakers are not reproduced.
    """
    base = current_table(played, teams).set_index("Team").loc[teams]
    lookup = {t: i for i, t in enumerate(teams)}
    p = np.tile(base.Pts.to_numpy(), (count, 1))
    gf = np.tile(base.GF.to_numpy(), (count, 1))
    ga = np.tile(base.GA.to_numpy(), (count, 1))
    rng = np.random.default_rng(seed)
    if len(remaining):
        hs = np.array([lookup[t] for t in remaining.HomeTeam], dtype=int)
        aw = np.array([lookup[t] for t in remaining.AwayTeam], dtype=int)
        gh = rng.poisson(mus[:, 0], size=(count, len(remaining)))
        ga_goals = rng.poisson(mus[:, 1], size=(count, len(remaining)))
        for j, (h, a) in enumerate(zip(hs, aw)):
            sh, sa = gh[:, j], ga_goals[:, j]
            gf[:, h] += sh; ga[:, h] += sa
            gf[:, a] += sa; ga[:, a] += sh
            p[:, h] += 3 * (sh > sa) + (sh == sa)
            p[:, a] += 3 * (sa > sh) + (sh == sa)
    order = np.lexsort((-gf, -(gf - ga), -p), axis=1)
    positions = np.empty_like(order)
    positions[np.arange(count)[:, None], order] = np.arange(1, len(teams) + 1)
    return pd.DataFrame({"Team": teams, "Projected pts": np.mean(p, axis=0).round(1),
                         "Title %": (100 * (positions == 1).mean(axis=0)).round(1),
                         "Top 4 %": (100 * (positions <= min(4, len(teams))).mean(axis=0)).round(1),
                         "Bottom 3 %": (100 * (positions > max(0, len(teams) - 3)).mean(axis=0)).round(1),
                         "Avg finish": np.mean(positions, axis=0).round(1)}).sort_values(
        ["Projected pts", "Title %"], ascending=False).reset_index(drop=True)


def ucl_simulate(entrants, date, hist, elo, last, model, count, seed):
    """Hypothetical 16-team single-elimination tournament, random bracket.

    Two leg ties through semifinals (home and away), single-match final.
    Aggregate ties resolve by a 50/50 penalty shootout, no extra time model.
    Fixtures/seedings are NOT the official historical UCL bracket.
    """
    rng = np.random.default_rng(seed)
    slots = np.asarray(entrants, dtype=object)
    rng.shuffle(slots)
    # Each possible home/away pairing predicted once from the SAME pre-cutoff snapshot.
    possible = [(a, b, date) for a in entrants for b in entrants if a != b]
    predicted = predict_many(possible, hist, elo, last, model)
    mu = {(a, b): predicted[i] for i, (a, b, _) in enumerate(possible)}
    initial = np.broadcast_to(slots, (count, len(slots))).copy()
    alive = initial
    stage_names = ["Round of 16", "Quarterfinals", "Semifinals", "Final"]
    stage_rates = {}
    example_rounds = []
    for round_idx, stage in enumerate(stage_names):
        stage_rates[stage] = {t: float((alive == t).any(axis=1).mean() * 100) for t in entrants}
        next_alive = np.empty((count, alive.shape[1] // 2), dtype=object)
        one_round = []
        for j in range(0, alive.shape[1], 2):
            left, right = alive[:, j], alive[:, j + 1]
            first, second = np.zeros(count, dtype=int), np.zeros(count, dtype=int)
            # Probabilities vary by participant pair as winners advance.
            pairs = sorted(set(zip(left.tolist(), right.tolist())))
            for a, b in pairs:
                sel = (left == a) & (right == b)
                n = int(sel.sum())
                h1, a1 = mu[(a, b)]
                g1 = rng.poisson(h1, n); g2 = rng.poisson(a1, n)
                if round_idx < 3:
                    h2, a2 = mu[(b, a)]
                    g3 = rng.poisson(h2, n); g4 = rng.poisson(a2, n)
                    first[sel] = g1 + g4
                    second[sel] = g2 + g3
                else:
                    first[sel] = g1
                    second[sel] = g2
            penalties = rng.random(count) < 0.5
            win_left = (first > second) | ((first == second) & penalties)
            next_alive[:, j // 2] = np.where(win_left, left, right)
            one_round.append((str(left[0]), str(right[0]), str(next_alive[0, j // 2]), int(first[0]), int(second[0])))
        example_rounds.append((stage, one_round))
        alive = next_alive
    champions = alive[:, 0]
    title = {t: float((champions == t).mean() * 100) for t in entrants}
    return pd.DataFrame({"Team": entrants, "Win trophy %": [round(title[t], 1) for t in entrants],
                         "Reach final %": [round(stage_rates["Final"][t], 1) for t in entrants],
                         "Reach semis %": [round(stage_rates["Semifinals"][t], 1) for t in entrants]}).sort_values(
        "Win trophy %", ascending=False).reset_index(drop=True), example_rounds


