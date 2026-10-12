"""Train match outcome and goal models on earlier seasons.

Check performance on a later historical season before refitting.
"""
from hashlib import sha256

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, log_loss

from train_model_v2 import make_examples
from train_model_v3 import add_goal_labels, train_logistic, make_poisson_models


@st.cache_resource(show_spinner="Training match models from historical results…")
def _train(_matches, fingerprint, target_season):
    X, meta = make_examples(_matches)
    goals = add_goal_labels(_matches, meta)
    historical = sorted(s for s in meta.Season.unique() if s < target_season)
    if len(historical) < 2:
        raise ValueError("At least two earlier seasons are needed to train and check the model")
    validation_season = historical[-1]
    earlier = meta.Season < validation_season
    validation = meta.Season == validation_season
    final_training = meta.Season < target_season
    if earlier.sum() < 100 or validation.sum() < 100:
        raise ValueError("Too few matches for chronological model evaluation")
    check_model = train_logistic(X.loc[earlier], meta.loc[earlier, "Result"])
    proba = check_model.predict_proba(X.loc[validation])
    actual = meta.loc[validation, "Result"]
    metrics = {
        "accuracy": float(accuracy_score(actual, check_model.classes_[proba.argmax(axis=1)])),
        "log_loss": float(log_loss(actual, proba, labels=list(check_model.classes_))),
        "test_matches": int(validation.sum()),
        "test_season": validation_season,
    }
    logistic = train_logistic(X.loc[final_training], meta.loc[final_training, "Result"])
    home_model, away_model = make_poisson_models(X.loc[final_training], goals[final_training.to_numpy()])
    return {
        "model": logistic, "poisson_home_model": home_model,
        "poisson_away_model": away_model, "features": list(X.columns),
        "test_metrics": metrics, "recent_matches": 5, "minimum_history": 3,
        "elo_k_factor": 20.0, "home_elo_advantage": 60.0,
        "training_through": historical[-1], "target_season": target_season,
    }


def load_real_models(matches, target_season="2026-27"):
    columns = ["Date", "Season", "Competition", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]
    history = matches.loc[matches.Season < target_season]
    digest = sha256(pd.util.hash_pandas_object(history[columns], index=False).values.tobytes()).hexdigest()
    return _train(history, digest, target_season)
