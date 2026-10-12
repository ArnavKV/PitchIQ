"""PitchIQ: experimental match outcome model v2.

Run from the PitchIQ repo: python train_model_v2.py
Keeps the existing match_predictor.joblib unchanged.
All features are computed from games played strictly BEFORE the match date.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "matches" / "all_matches.csv"
OUTPUT = ROOT / "models" / "match_predictor_v2.joblib"
TEST_SEASON = "2023-24"
RECENT = 5
MIN_PAST = 3
CLASSES = ["H", "D", "A"]
K_FACTOR = 20.0
HOME_ELO_ADVANTAGE = 60.0


def rolling_form(games, side=None):
    if side is not None:
        games = [g for g in games if g["side"] == side]
    recent = games[-RECENT:]
    if not recent:
        return [np.nan] * 4
    n = len(recent)
    return [
        sum(g["points"] for g in recent) / n,
        sum(g["gf"] for g in recent) / n,
        sum(g["ga"] for g in recent) / n,
        sum(g["points"] == 3 for g in recent) / n,
    ]


def make_examples(df):
    feature_rows = []
    meta_rows = []
    # Separate league strength histories; same team may appear in UCL and domestic leagues.
    for competition, group in df.groupby("Competition", sort=False):
        group = group.sort_values(["Date", "HomeTeam", "AwayTeam"])
        history = defaultdict(list)
        elo = defaultdict(lambda: 1500.0)
        last_played = {}
        # Generate ALL same-date feature rows before updating histories.
        for date, day in group.groupby("Date", sort=True):
            for row in day.itertuples(index=False):
                h, a = str(row.HomeTeam), str(row.AwayTeam)
                hh, ah = history[h], history[a]
                if len(hh) < MIN_PAST or len(ah) < MIN_PAST:
                    continue
                home_all = rolling_form(hh)
                away_all = rolling_form(ah)
                home_at_home = rolling_form(hh, "H")
                away_at_away = rolling_form(ah, "A")
                home_rest = (date - last_played[h]).days if h in last_played else np.nan
                away_rest = (date - last_played[a]).days if a in last_played else np.nan
                values = (
                    home_all + away_all + home_at_home + away_at_away +
                    [elo[h] - elo[a], elo[h], elo[a], home_rest, away_rest]
                )
                feature_rows.append(values)
                meta_rows.append({
                    "Season": row.Season,
                    "Date": date,
                    "Competition": competition,
                    "Result": row.Result,
                })

            for row in day.itertuples(index=False):
                h, a = str(row.HomeTeam), str(row.AwayTeam)
                hg, ag = int(row.HomeGoals), int(row.AwayGoals)
                hp = 3 if hg > ag else (1 if hg == ag else 0)
                ap = 3 if ag > hg else (1 if ag == hg else 0)
                eh, ea = elo[h], elo[a]
                expected_home = 1 / (1 + 10 ** ((ea - (eh + HOME_ELO_ADVANTAGE)) / 400))
                actual_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
                change = K_FACTOR * (actual_home - expected_home)
                elo[h], elo[a] = eh + change, ea - change
                history[h].append({"points": hp, "gf": hg, "ga": ag, "side": "H"})
                history[a].append({"points": ap, "gf": ag, "ga": hg, "side": "A"})
                last_played[h] = date
                last_played[a] = date
    columns = (
        [f"HomeRecent_{x}" for x in ["Points", "Scored", "Conceded", "WinRate"]] +
        [f"AwayRecent_{x}" for x in ["Points", "Scored", "Conceded", "WinRate"]] +
        [f"HomeAtHome_{x}" for x in ["Points", "Scored", "Conceded", "WinRate"]] +
        [f"AwayAtAway_{x}" for x in ["Points", "Scored", "Conceded", "WinRate"]] +
        ["EloDifference", "HomeElo", "AwayElo", "HomeDaysRest", "AwayDaysRest"]
    )
    return pd.DataFrame(feature_rows, columns=columns), pd.DataFrame(meta_rows)


def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)

    # Scikit-learn expects alphabetical class order: A, D, H
    correct_order = sorted(CLASSES)

    index = {c: i for i, c in enumerate(model.classes_)}

    ordered = np.column_stack([
        probs[:, index[c]] for c in correct_order
    ])

    predictions = np.array(correct_order)[
        ordered.argmax(axis=1)
    ]

    draw_mask = y_test.to_numpy() == "D"

    draw_recall = (
        float(np.mean(predictions[draw_mask] == "D"))
        if draw_mask.any()
        else float("nan")
    )

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "log_loss": log_loss(
            y_test,
            ordered,
            labels=correct_order
        ),
        "draw_recall": draw_recall,
        "predictions": predictions,
    }

def main():
    if not INPUT.exists():
        raise SystemExit(f"Missing file: {INPUT}")
    df = pd.read_csv(INPUT)
    required = {"Date", "Season", "Competition", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "Result"}
    if missing := required - set(df.columns):
        raise SystemExit(f"Missing CSV columns: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["HomeGoals"] = pd.to_numeric(df["HomeGoals"], errors="coerce")
    df["AwayGoals"] = pd.to_numeric(df["AwayGoals"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "Season", "Competition"])
    df = df[df["Result"].isin(CLASSES)].copy()
    # Avoid processing the same match twice if a source contains duplicates.
    df = df.drop_duplicates(subset=["Date", "Competition", "HomeTeam", "AwayTeam"])
    df["Season"] = df["Season"].astype(str)
    print("=== PitchIQ Match Predictor V2 ===")
    print(f"Matches loaded: {len(df):,}")
    X, meta = make_examples(df)
    y = meta["Result"]
    train_mask = meta["Season"] != TEST_SEASON
    test_mask = meta["Season"] == TEST_SEASON
    if not test_mask.any() or not train_mask.any():
        raise SystemExit(f"Expected held-out season {TEST_SEASON} and earlier seasons. Found {sorted(meta.Season.unique())}")
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    print(f"Training examples: {len(X_train):,}; test examples: {len(X_test):,} ({TEST_SEASON})")
    baseline_accuracy = y_test.value_counts(normalize=True).max()
    print(f"Majority-class baseline accuracy: {baseline_accuracy:.2%}")
    print("Test class counts:", y_test.value_counts().to_dict())
    candidates = {
        "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000)),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=250, min_samples_leaf=12, max_features="sqrt", random_state=42, n_jobs=-1)),
        "Gradient Boosting": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, l2_regularization=10.0, random_state=42)),
    }
    results = {}
    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = (model, metrics)
        print(f"  Accuracy: {metrics['accuracy']:.2%} | Log loss: {metrics['log_loss']:.4f} | Draw recall: {metrics['draw_recall']:.2%}")

    # Choose the best held-out log loss as a provisional winner, not a final unbiased score.
    winner = min(results, key=lambda name: results[name][1]["log_loss"])
    best_model, best_metrics = results[winner]
    print(f"\nBest held-out log loss: {winner}")
    print("Confusion matrix (actual rows, predicted columns; H/D/A):")
    print(confusion_matrix(y_test, best_metrics["predictions"], labels=CLASSES))
    print(classification_report(y_test, best_metrics["predictions"], labels=CLASSES, target_names=["Home Win", "Draw", "Away Win"], zero_division=0))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": best_model,
        "features": list(X.columns),
        "test_season": TEST_SEASON,
        "recent_matches": RECENT,
        "minimum_history": MIN_PAST,
        "elo_k_factor": K_FACTOR,
        "home_elo_advantage": HOME_ELO_ADVANTAGE,
        "best_model": winner,
        "test_metrics": {k: v for k, v in best_metrics.items() if k != "predictions"},
        "notes": "Experimental. V2 features and inference logic differ from the existing Streamlit predictor.",
    }, OUTPUT)
    print(f"\nSaved experimental model: {OUTPUT}")
    print("Your original models/match_predictor.joblib has NOT been modified.")
    print("CAUTION: Selecting a model on this test set makes the winner's score optimistic. Validate again on a new season before deployment.")

if __name__ == "__main__":
    main()
