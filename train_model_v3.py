"""PitchIQ V3: Poisson scoreline model versus V2 logistic baseline.

Run: python train_model_v3.py
Requires train_model_v2.py next to this file and data/matches/all_matches.csv.
Does not modify either existing model. Uses a 2022-23 validation season and
an untouched 2023-24 final test season. All features are pre-match features.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import skellam
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_model_v2 import make_examples, CLASSES

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "matches" / "all_matches.csv"
OUTPUT = ROOT / "models" / "match_predictor_v3.joblib"
VALIDATION_SEASON = "2022-23"
TEST_SEASON = "2023-24"
CLASS_ORDER = sorted(CLASSES)  # A, D, H, required by log_loss


def load_matches():
    data = pd.read_csv(INPUT)
    required = {"Date", "Season", "Competition", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "Result"}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for name in ("HomeGoals", "AwayGoals"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data = data.dropna(subset=["Date", "Season", "Competition", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"])
    data = data[data.Result.isin(CLASSES)].copy()
    data = data.drop_duplicates(subset=["Date", "Competition", "HomeTeam", "AwayTeam"])
    data["Season"] = data["Season"].astype(str)
    return data


def add_goal_labels(matches, meta):
    """Align exact match results to v2's pre-match feature rows.

    V2's make_examples omits team names, so we use a match key for game results:
    competition + date + outcome. This is ambiguous when two games have the
    same result on the same day! The robust solution is adding scores directly
    inside make_examples; we reconstruct them below with a validated row order.
    """
    # Reconstruct corresponding rows using the SAME competition/date/home/away
    # iteration as make_examples. Counts reproduce its MIN_PAST eligibility.
    from collections import defaultdict
    from train_model_v2 import MIN_PAST
    records = []
    for competition, group in matches.groupby("Competition", sort=False):
        group = group.sort_values(["Date", "HomeTeam", "AwayTeam"])
        counts = defaultdict(int)
        for date, day in group.groupby("Date", sort=True):
            for match in day.itertuples(index=False):
                h, a = str(match.HomeTeam), str(match.AwayTeam)
                if counts[h] >= MIN_PAST and counts[a] >= MIN_PAST:
                    records.append((competition, date, match.Result, int(match.HomeGoals), int(match.AwayGoals)))
            for match in day.itertuples(index=False):
                counts[str(match.HomeTeam)] += 1
                counts[str(match.AwayTeam)] += 1
    if len(records) != len(meta):
        raise RuntimeError(f"Score rows ({len(records)}) != feature rows ({len(meta)})")
    # Verify order/identity so scores cannot silently land on wrong feature rows.
    for index, ((comp, date, result, _, _), row) in enumerate(zip(records, meta.itertuples(index=False))):
        if comp != row.Competition or date != row.Date or result != row.Result:
            raise RuntimeError(f"Feature-score ordering mismatch at row {index}")
    return np.asarray([(r[3], r[4]) for r in records], dtype=int)


def make_poisson_models(X_train, goals_train):
    models = []
    for side in (0, 1):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PoissonRegressor(alpha=1.0, max_iter=500),
        )
        model.fit(X_train, goals_train[:, side])
        models.append(model)
    return models


def poisson_outcomes(models, X):
    home_mu = np.clip(models[0].predict(X), 0.05, 8.0)
    away_mu = np.clip(models[1].predict(X), 0.05, 8.0)
    draw = skellam.pmf(0, home_mu, away_mu)
    away = skellam.cdf(-1, home_mu, away_mu)
    home = 1.0 - draw - away
    # CLASS_ORDER = A, D, H
    probs = np.column_stack([away, draw, home])
    probs = np.clip(probs, 1e-12, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, home_mu, away_mu


def logistic_outcomes(model, X):
    raw = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in CLASS_ORDER]
    return raw[:, order]


def summarize(title, truth, probs):
    predictions = np.asarray(CLASS_ORDER)[np.argmax(probs, axis=1)]
    draw_mask = np.asarray(truth) == "D"
    accuracy = accuracy_score(truth, predictions)
    loss = log_loss(truth, probs, labels=CLASS_ORDER)
    recall = np.mean(predictions[draw_mask] == "D") if draw_mask.any() else float("nan")
    print(f"  {title:<20} accuracy={accuracy:.2%}  log loss={loss:.4f}  draw recall={recall:.2%}")
    return {"accuracy": float(accuracy), "log_loss": float(loss), "draw_recall": float(recall)}, predictions


def train_logistic(X, y):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000),
    )
    model.fit(X, y)
    return model


def main():
    if not INPUT.exists():
        raise SystemExit(f"Missing match data: {INPUT}")
    matches = load_matches()
    X, meta = make_examples(matches)
    goals = add_goal_labels(matches, meta)
    seasons = sorted(meta.Season.unique())
    print("=== PitchIQ Match Predictor V3: Poisson Goals ===")
    print(f"Matches loaded: {len(matches):,}; eligible examples: {len(X):,}")
    print("Available seasons:", ", ".join(seasons))
    if VALIDATION_SEASON not in seasons or TEST_SEASON not in seasons:
        raise ValueError("Missing validation or test season")
    early = meta.Season < VALIDATION_SEASON
    valid = meta.Season == VALIDATION_SEASON
    final_train = meta.Season < TEST_SEASON
    test = meta.Season == TEST_SEASON
    if not (early.any() and valid.any() and test.any()):
        raise ValueError("Expected multiple historical seasons")
    y = meta.Result
    print(f"Validation: {sum(early):,} earlier examples -> {sum(valid):,} {VALIDATION_SEASON} examples")
    val_logistic = train_logistic(X.loc[early], y.loc[early])
    val_poisson = make_poisson_models(X.loc[early], goals[early.to_numpy()])
    val_l_probs = logistic_outcomes(val_logistic, X.loc[valid])
    val_p_probs, _, _ = poisson_outcomes(val_poisson, X.loc[valid])
    print("\nValidation (for model selection):")
    val_l, _ = summarize("V2-style Logistic", y.loc[valid], val_l_probs)
    val_p, _ = summarize("V3 Poisson", y.loc[valid], val_p_probs)
    chosen = "poisson" if val_p["log_loss"] < val_l["log_loss"] else "logistic"
    print(f"Selected by validation log loss: {chosen}")

    print(f"\nFinal test: {sum(final_train):,} historical examples -> {sum(test):,} {TEST_SEASON} examples")
    logistic = train_logistic(X.loc[final_train], y.loc[final_train])
    poisson = make_poisson_models(X.loc[final_train], goals[final_train.to_numpy()])
    log_probs = logistic_outcomes(logistic, X.loc[test])
    pois_probs, home_mu, away_mu = poisson_outcomes(poisson, X.loc[test])
    print("\nFinal held-out test (not used for selection):")
    test_l, pred_l = summarize("V2-style Logistic", y.loc[test], log_probs)
    test_p, pred_p = summarize("V3 Poisson", y.loc[test], pois_probs)
    print(f"\nSelected model: {chosen}")
    selected_predictions = pred_p if chosen == "poisson" else pred_l
    print("Confusion matrix of selected model (actual rows, predicted columns: H, D, A):")
    print(confusion_matrix(y.loc[test], selected_predictions, labels=CLASSES))
    print(f"\nExample predicted home/away goals for first 3 test matches:")
    for i in range(min(3, len(home_mu))):
        print(f"  Home {home_mu[i]:.2f} – Away {away_mu[i]:.2f} | H={pois_probs[i, 2]:.1%}, D={pois_probs[i, 1]:.1%}, A={pois_probs[i, 0]:.1%}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "selected_model": chosen,
        "poisson_home_model": poisson[0],
        "poisson_away_model": poisson[1],
        "logistic_model": logistic,
        "features": list(X.columns),
        "classes": CLASS_ORDER,
        "validation_season": VALIDATION_SEASON,
        "test_season": TEST_SEASON,
        "validation_metrics": {"logistic": val_l, "poisson": val_p},
        "test_metrics": {"logistic": test_l, "poisson": test_p},
        "notes": "Experimental V3. Reuses V2 prematch features; not compatible with current Streamlit predictor without UI changes.",
    }, OUTPUT)
    print(f"\nSaved experimental model: {OUTPUT}")
    print("Existing V1 and V2 model files have NOT been changed.")
    print("Poisson assumes independent home and away goal counts; later we can test a draw correction and probability calibration.")


if __name__ == "__main__":
    main()
