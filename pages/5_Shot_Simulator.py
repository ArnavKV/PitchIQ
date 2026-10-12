"""PitchIQ: animated 2D shot simulator (view from behind the shooter).

Uses the shared xG model and profiles without modifying their artifacts.
"""
import json
import streamlit as st

from pitchiq.data import load_shots, load_model as load_artifact
from pitchiq.shots import shot_prediction
from pitchiq.ui import ROOT, configure_page, footer, page_header

configure_page("Shot simulator")
page_header("Shot simulator", "Pick your shooter. Set the chance. Aim for the corner and watch your attempt play out.", "Simulate / On the pitch")


def load_model():
    return load_artifact("xg_predictor.joblib", ("model", "features"))


try:
    shots = load_shots()
    info = load_model()
except (OSError, ValueError, KeyError, ImportError) as exc:
    st.error("Shot data or trained model could not be loaded.")
    st.code(str(exc))
    st.stop()

names = sorted(shots["Shooter"].dropna().astype(str).unique())
if not names:
    st.error("The shot dataset contains no shooters.")
    st.stop()
def_shooter = next((i for i, n in enumerate(names) if "Messi" in n), 0)
keeper_profiles = info.get("goalkeeper_factors", {})
keepers = ["Average / unspecified goalkeeper"] + sorted(keeper_profiles.keys())
def_keeper = next((i for i, n in enumerate(keepers) if "Courtois" in n), 0)

left, right = st.columns(2)
with left:
    shooter = st.selectbox("Shooter", names, index=def_shooter)
with right:
    keeper = st.selectbox("Goalkeeper", keepers, index=def_keeper)
    kinds_available = sorted(shots["ShotType"].dropna().astype(str).unique())
    kinds = [x for x in kinds_available if x != "Corner"]
    default_kind = kinds.index("Open Play") if "Open Play" in kinds else 0
    kind = st.selectbox("Shot type", kinds, index=default_kind)

# Only open-play shots can be headers. Widget key changes when allowed
# options change, so a previously selected header cannot remain active.
allowed_parts = (["Left Foot", "Right Foot", "Head"]
                 if kind == "Open Play" else ["Left Foot", "Right Foot"])
with left:
    foot = st.selectbox("Body part", allowed_parts,
                        key=f"sim_body_part_{kind}")
if kind != "Open Play" and foot == "Head":
    st.error("Headers are not valid for this shot type.")
    st.stop()

is_penalty = kind == "Penalty"
if is_penalty:
    st.info("Penalty selected: shot location fixed to the penalty spot (108, 40).")
xcol, ycol, dcol = st.columns(3)
with xcol:
    sx = st.slider("Shot X (toward goal →)", 60, 118, 102, disabled=is_penalty)
with ycol:
    sy = st.slider("Shot Y (center = 40)", 0, 80, 40, disabled=is_penalty)
with dcol:
    target = st.selectbox("Aim the shot", ["Left corner", "Center", "Right corner"])
if is_penalty:
    sx, sy = 108, 40

subset = shots.loc[shots["Shooter"] == shooter]
prediction = shot_prediction(info, subset, keeper, sx, sy, foot, kind)
standard, probability = prediction["standard"], prediction["probability"]
distance = prediction["distance"]

m1, m2, m3 = st.columns(3)
m1.metric("Estimated chance of scoring", f"{probability:.2%}")
m2.metric("Standard xG", f"{standard:.3f}")
m3.metric("Shot distance (pitch units)", f"{distance:.1f}")
st.caption("The selected aim direction controls the animation, not the xG probability. Keeper adjustment is neutral for penalties, as in your current model.")

# Safe injection: serialize everything to JS using JSON, never interpolate into JS strings.
settings = json.dumps({
    "shooter": shooter, "keeper": keeper,
    "probability": probability, "target": target,
    "penalty": is_penalty, "distance": distance,
    "shot_x": sx, "shot_y": sy, "body_part": foot,
    "jersey": "7" if "Cristiano Ronaldo" in shooter else "10" if "Messi" in shooter else "9",
}, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")

# The reviewed animation template contains no user-authored executable HTML.
page = (ROOT / "assets/shot_simulator.html").read_text().replace("__SETTINGS__", settings)

st.iframe(page, height=660, alt="Interactive shot simulation")


footer("Shot data: StatsBomb Open Data · Attempts are sampled from the xG estimate; animation is illustrative.")
