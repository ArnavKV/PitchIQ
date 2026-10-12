
from pathlib import Path
from collections import defaultdict
import joblib

import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss
)


# Settings
BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR / "data" / "matches" / "all_matches.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "match_predictor.joblib"

TRAIN_SEASONS = [
    "2020-21",
    "2021-22",
    "2022-23"
]

TEST_SEASON = "2023-24"

RECENT_MATCHES = 5
MIN_HISTORY = 3

FEATURE_COLUMNS = [
    "HomePointsAverage",
    "AwayPointsAverage",
    "HomeGoalsScoredAverage",
    "AwayGoalsScoredAverage",
    "HomeGoalsConcededAverage",
    "AwayGoalsConcededAverage",
    "HomeWinRate",
    "AwayWinRate"
]


# Load match data
def load_data():
    df = pd.read_csv(DATA_FILE)

    required = [
        "Date",
        "Season",
        "Competition",
        "HomeTeam",
        "AwayTeam",
        "HomeGoals",
        "AwayGoals",
        "Result"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        ["Date", "Competition", "HomeTeam"]
    ).reset_index(drop=True)

    return df


# Team performance
def team_features(history):
    recent = history[-RECENT_MATCHES:]

    if not recent:
        return None

    count = len(recent)

    points = sum(
        match["points"] for match in recent
    )

    goals_scored = sum(
        match["scored"] for match in recent
    )

    goals_conceded = sum(
        match["conceded"] for match in recent
    )

    wins = sum(
        match["points"] == 3 for match in recent
    )

    return {
        "points": points / count,
        "scored": goals_scored / count,
        "conceded": goals_conceded / count,
        "win_rate": wins / count
    }


# Build pre-match features
def build_features(df):
    # A separate history for every team,
    # season, and competition.
    histories = defaultdict(list)

    records = []

    # Process one calendar date at a time.
    # All games from the same date receive
    # their features before any results from
    # that date are added to team histories.
    for date, daily_games in df.groupby(
        "Date", sort=True
    ):
        daily_updates = []

        for _, match in daily_games.iterrows():
            season = match["Season"]
            competition = match["Competition"]

            home = match["HomeTeam"]
            away = match["AwayTeam"]

            home_key = (
                season, competition, home
            )

            away_key = (
                season, competition, away
            )

            home_history = histories[home_key]
            away_history = histories[away_key]

            # Require some previous matches
            # before making a prediction.
            if (
                len(home_history) >= MIN_HISTORY
                and len(away_history) >= MIN_HISTORY
            ):
                home_stats = team_features(
                    home_history
                )

                away_stats = team_features(
                    away_history
                )

                record = {
                    "Date": date,
                    "Season": season,
                    "Competition": competition,
                    "HomeTeam": home,
                    "AwayTeam": away,

                    "HomePointsAverage":
                        home_stats["points"],

                    "AwayPointsAverage":
                        away_stats["points"],

                    "HomeGoalsScoredAverage":
                        home_stats["scored"],

                    "AwayGoalsScoredAverage":
                        away_stats["scored"],

                    "HomeGoalsConcededAverage":
                        home_stats["conceded"],

                    "AwayGoalsConcededAverage":
                        away_stats["conceded"],

                    "HomeWinRate":
                        home_stats["win_rate"],

                    "AwayWinRate":
                        away_stats["win_rate"],

                    "Result": match["Result"]
                }

                records.append(record)

            home_goals = int(match["HomeGoals"])
            away_goals = int(match["AwayGoals"])

            if home_goals > away_goals:
                home_points = 3
                away_points = 0

            elif home_goals < away_goals:
                home_points = 0
                away_points = 3

            else:
                home_points = 1
                away_points = 1

            # Save updates until the whole
            # day's features have been created.
            daily_updates.append((
                home_key,
                {
                    "points": home_points,
                    "scored": home_goals,
                    "conceded": away_goals
                }
            ))

            daily_updates.append((
                away_key,
                {
                    "points": away_points,
                    "scored": away_goals,
                    "conceded": home_goals
                }
            ))

        for team_key, result in daily_updates:
            histories[team_key].append(result)

    return pd.DataFrame(records)


# Train and evaluate model
def main():
    print("================================")
    print("       PITCHIQ MATCH AI")
    print("================================")

    print("\nLoading historical matches...")

    df = load_data()

    print(f"Total matches: {len(df)}")

    print("\nCalculating pre-match statistics...")

    features_df = build_features(df)

    if features_df.empty:
        raise ValueError(
            "No training examples were generated."
        )

    print(
        f"Usable examples: {len(features_df)}"
    )

    # Split by season.
    train_df = features_df[
        features_df["Season"].isin(
            TRAIN_SEASONS
        )
    ].copy()

    test_df = features_df[
        features_df["Season"] == TEST_SEASON
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Training or testing data is empty."
        )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["Result"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Result"]

    print("\nDataset split:")
    print(f"Training matches: {len(X_train)}")
    print(f"Testing matches: {len(X_test)}")

    # Baseline: always predict the most
    # frequent training outcome.
    most_common = y_train.mode().iloc[0]

    baseline_predictions = [
        most_common
    ] * len(y_test)

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions
    )

    # Logistic regression classification
    # model with standardized features.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000
        )
    )

    print("\nTraining model...")

    model.fit(X_train, y_train)

    # Predictions on unseen 2023-24 games.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(
        X_test
    )

    model_accuracy = accuracy_score(
        y_test,
        predictions
    )

    model_log_loss = log_loss(
        y_test,
        probabilities,
        labels=model.classes_
    )

    print("\n================================")
    print("          RESULTS")
    print("================================")

    print(
        f"Baseline accuracy: "
        f"{baseline_accuracy:.2%}"
    )

    print(
        f"Model accuracy:    "
        f"{model_accuracy:.2%}"
    )

    print(
        f"Log loss:          "
        f"{model_log_loss:.3f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["H", "D", "A"],
            target_names=[
                "Home Win",
                "Draw",
                "Away Win"
            ],
            zero_division=0
        )
    )

    # Save the model and feature names.
    # Only load this file later if it was
    # created by your own trusted script.
    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "training_seasons": TRAIN_SEASONS,
            "test_season": TEST_SEASON,
            "recent_matches": RECENT_MATCHES,
            "minimum_history": MIN_HISTORY
        },
        MODEL_FILE
    )

    print(
        f"\nModel saved to: {MODEL_FILE}"
    )


if __name__ == "__main__":
    main()
