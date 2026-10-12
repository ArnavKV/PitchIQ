"""Load the fictional demo or a complete set of local research assets."""
from pathlib import Path
import os

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MATCHES_PATH = ROOT / "data/matches/all_matches.csv"
SHOTS_PATH = ROOT / "data/shots/shots.csv"
MODELS_PATH = ROOT / "models"
PLAYER_PATHS = (
    ROOT / "data/player_comparison/cleaned/transfermarkt_clean.csv",
    ROOT / "data/player_comparison/cleaned/understat_clean.csv",
)


def using_demo():
    """Use one backend for all tools; never mix synthetic and real assets."""
    mode = os.environ.get("PITCHIQ_DATA_MODE", "auto").strip().lower()
    if mode not in {"auto", "demo", "local"}:
        raise ValueError("PITCHIQ_DATA_MODE must be auto, demo, or local.")
    if mode != "auto":
        return mode == "demo"
    required = [MATCHES_PATH, SHOTS_PATH, *PLAYER_PATHS]
    required += [MODELS_PATH / name for name in (
        "match_predictor_v2.joblib", "match_predictor_v3.joblib", "xg_predictor.joblib",
    )]
    return not all(path.is_file() for path in required)


def player_data_versions():
    return ("synthetic-v1", "synthetic-v1") if using_demo() else tuple(path.stat().st_mtime_ns for path in PLAYER_PATHS)


@st.cache_data(show_spinner="Loading player seasons…")
def _read_players(paths, versions):
    return tuple(pd.read_csv(path, low_memory=False) for path in paths)


def load_player_frames():
    if using_demo():
        from pitchiq.demo import player_frames
        return player_frames()
    return _read_players(tuple(str(path) for path in PLAYER_PATHS), player_data_versions())


def validate_matches(frame):
    required = {"Date", "Season", "Competition", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "Result"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing match columns: {sorted(missing)}")
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in ("HomeGoals", "AwayGoals"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
    frame = frame[frame["Result"].isin(["H", "D", "A"])]
    for column in ("HomeGoals", "AwayGoals"):
        frame = frame.loc[frame[column].ge(0) & frame[column].mod(1).eq(0)].copy()
        frame[column] = frame[column].astype(int)
    frame["Season"] = frame["Season"].astype(str)
    return frame.drop_duplicates(["Date", "Competition", "HomeTeam", "AwayTeam"]).sort_values(
        ["Date", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)


@st.cache_data(show_spinner="Loading match history…")
def _read_matches(path, version):
    return validate_matches(pd.read_csv(path))


def load_matches():
    if using_demo():
        from pitchiq.demo import matches
        return validate_matches(matches())
    return _read_matches(str(MATCHES_PATH), MATCHES_PATH.stat().st_mtime_ns)


@st.cache_data(show_spinner="Loading shot data…")
def _read_shots(path, version):
    frame = pd.read_csv(path)
    required = {"Shooter", "BodyPart", "ShotType", "Goal", "StatsBomb_xG"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing shot columns: {sorted(missing)}")
    frame["Goal"] = pd.to_numeric(frame["Goal"], errors="coerce")
    frame["StatsBomb_xG"] = pd.to_numeric(frame["StatsBomb_xG"], errors="coerce")
    return frame.dropna(subset=["Shooter", "BodyPart", "ShotType", "Goal"])


def load_shots():
    if using_demo():
        from pitchiq.demo import shots
        return shots()
    return _read_shots(str(SHOTS_PATH), SHOTS_PATH.stat().st_mtime_ns)


@st.cache_resource(show_spinner="Loading trained model…")
def _read_model(path, version):
    # joblib can execute code: use only locally trained, trusted artifacts.
    return joblib.load(path)


def load_model(filename, required=()):
    if using_demo():
        from pitchiq.demo import model
        artifact = model(filename)
    else:
        path = MODELS_PATH / filename
        artifact = _read_model(str(path), path.stat().st_mtime_ns)
    if not isinstance(artifact, dict) or set(required) - set(artifact):
        raise ValueError(f"Incompatible {filename}; expected keys {sorted(required)}.")
    return artifact
