"""Shot geometry and profile adjustments shared by both interactive tools."""
import math

import pandas as pd


def shot_geometry(x, y):
    dx, dy = 120.0 - float(x), 40.0 - float(y)
    left, right = 36.0 - float(y), 44.0 - float(y)
    distance = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(abs(dx * (right - left)), dx * dx + left * right))
    return distance, angle


def adjust_probability(probability, multiplier):
    if not 0 <= probability <= 1 or not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Probability must be in [0, 1] and the odds factor must be positive.")
    return probability * multiplier / (1 - probability + probability * multiplier)


def shot_prediction(info, player_shots, keeper, x, y, body_part, shot_type):
    distance, angle = shot_geometry(x, y)
    features = pd.DataFrame([{
        "ShotDistance": distance, "ShotAngle": angle,
        "BodyPart": body_part, "ShotType": shot_type,
    }])[info["features"]]
    model = info["model"]
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The xG model does not have a scoring (1) class.")
    standard = float(model.predict_proba(features)[0, classes.index(1)])
    ids = pd.to_numeric(player_shots.get("ShooterID", pd.Series(dtype=float)), errors="coerce").dropna()
    pid = str(int(ids.mode().iloc[0])) if not ids.empty else ""
    profile = info.get("player_factors", {}).get(pid, {})
    foot_profile = info.get("player_foot_factors", {}).get(f"{pid}|{body_part}")
    multiplier = float((foot_profile or profile).get("multiplier", 1.0))
    keeper_multiplier = float(info.get("goalkeeper_factors", {}).get(keeper, {}).get("multiplier", 1.0)) if shot_type != "Penalty" else 1.0
    return {
        "standard": standard,
        "player_probability": adjust_probability(standard, multiplier),
        "probability": adjust_probability(standard, multiplier * keeper_multiplier),
        "distance": distance, "angle": angle, "profile": profile,
        "foot_profile": foot_profile, "multiplier": multiplier,
        "keeper_multiplier": keeper_multiplier,
    }
