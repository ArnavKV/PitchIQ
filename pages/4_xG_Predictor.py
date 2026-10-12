import pandas as pd
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import streamlit as st

from pitchiq.data import load_shots, load_model as load_artifact, using_demo
from pitchiq.shots import shot_prediction
from pitchiq.ui import configure_page, footer, page_header

configure_page("xG predictor")
page_header("xG predictor", "Move the shot. Change the angle. Discover what makes a chance worth taking.", "Predict / Shot quality")


def load_model():
    return load_artifact("xg_predictor.joblib", ("model", "features"))


try:
    shots = load_shots()
    model_info = load_model()
except (OSError, ValueError, KeyError, ImportError) as error:
    st.error("Couldn't load the xG dataset or trained model.")
    st.code(str(error))
    st.stop()

# Store the shot location directly in slider widget state so that
# both pitch clicks and slider adjustments update the same coordinates.
if "pitchiq_shot_x" not in st.session_state:
    st.session_state.pitchiq_shot_x = 102
if "pitchiq_shot_y" not in st.session_state:
    st.session_state.pitchiq_shot_y = 40

st.subheader("1. Choose a shooter")
player_names = sorted(shots["Shooter"].dropna().astype(str).unique().tolist())
if not player_names:
    st.error("No shooters found in shots.csv")
    st.stop()
initial_player = next((i for i, name in enumerate(player_names) if "Messi" in name), 0)
selected_player = st.selectbox("Shooter", player_names, index=initial_player)
player_shots = shots[shots["Shooter"] == selected_player].copy()
player_shots["StatsBomb_xG"] = pd.to_numeric(player_shots["StatsBomb_xG"], errors="coerce")
player_reference_xg = player_shots["StatsBomb_xG"].sum(min_count=1)
a, b, c = st.columns(3)
a.metric("Recorded shots", f"{len(player_shots):,}")
b.metric("Recorded goals", f"{int(player_shots['Goal'].sum()):,}")
c.metric("Synthetic reference xG" if using_demo() else "StatsBomb total xG", f"{player_reference_xg:.2f}" if pd.notna(player_reference_xg) else "N/A")
st.caption("Figures cover generated fictional chances." if using_demo() else "Figures cover downloaded StatsBomb matches, not complete career totals.")

st.subheader("2. Choose a goalkeeper")
keeper_profiles = model_info.get("goalkeeper_factors", {})
if keeper_profiles:
    available_keepers = sorted(keeper_profiles.keys())
    preferred = next((i for i, name in enumerate(available_keepers) if "Neuer" in name), 0)
    selected_goalkeeper = st.selectbox(
        "Goalkeeper (historical shot-stopping profile)",
        ["Average / unspecified goalkeeper"] + available_keepers,
        index=preferred + 1,
    )
    if selected_goalkeeper in keeper_profiles:
        keeper_info = keeper_profiles[selected_goalkeeper]
        c1, c2, c3 = st.columns(3)
        c1.metric("Shots faced in training data", f"{keeper_info['shots']:,}")
        c2.metric("Goals conceded (sample)", f"{keeper_info['goals_conceded']:,}")
        c3.metric("Reference xG against", f"{keeper_info['reference_xg_against']:.1f}")
    st.caption("Keeper profiles use identifiable opposing keepers only, from training matches. The dropdown can simulate hypothetical matchups, not just real recorded ones.")
else:
    selected_goalkeeper = "Average / unspecified goalkeeper"
    if "Goalkeeper" not in shots.columns:
        st.warning("Goalkeeper names missing from shots.csv. Run the latest prepare_shots.py to rebuild from cached data, then retrain.")
    else:
        st.warning("No goalkeeper had enough identifiable shots for a reliable training profile. See terminal training coverage, and consider expanding data.")

st.subheader("3. Shot details")
body_parts = sorted(shots["BodyPart"].dropna().astype(str).unique().tolist())
shot_types = [
    name for name in sorted(shots["ShotType"].dropna().astype(str).unique().tolist())
    if name != "Corner"
]
left, right = st.columns(2)
with left:
    body_part = st.selectbox("Body part", body_parts, index=body_parts.index("Right Foot") if "Right Foot" in body_parts else 0)
with right:
    shot_type = st.selectbox("Shot type", shot_types, index=shot_types.index("Open Play") if "Open Play" in shot_types else 0)

st.subheader("4. Choose a shot location")
is_penalty = shot_type == "Penalty"

# The clickable-image component gets unmounted during penalties. When
# switching back, give it a fresh key to ensure the pitch mounts again.
if "pitchiq_previous_penalty" not in st.session_state:
    st.session_state.pitchiq_previous_penalty = is_penalty
if "pitchiq_pitch_version" not in st.session_state:
    st.session_state.pitchiq_pitch_version = 0
if st.session_state.pitchiq_previous_penalty != is_penalty:
    st.session_state.pitchiq_previous_penalty = is_penalty
    st.session_state.pitchiq_pitch_version += 1
    st.session_state.pitchiq_last_click = None
if is_penalty:
    st.info("Penalty selected — shot locked to the penalty spot (108, 40). Change the shot type to move it.")
else:
    st.write("**Click anywhere on the green pitch** to place a shot. The orange marker shows the chosen location.")

# Click anywhere on the pitch image (not just on dots).
# The image is 840 x 560 pixels and has a 24-pixel margin.
# Coordinates: x = 60..120, y = 0..80 (StatsBomb orientation).
WIDTH, HEIGHT, MARGIN = 840, 560, 24
LEFT, TOP = MARGIN, MARGIN
RIGHT, BOTTOM = WIDTH - MARGIN, HEIGHT - MARGIN

def pitch_to_pixels(x, y):
    px = LEFT + (float(x) - 60.0) / 60.0 * (RIGHT - LEFT)
    py = TOP + float(y) / 80.0 * (BOTTOM - TOP)
    return round(px), round(py)

def pixels_to_pitch(px, py, display_width=WIDTH, display_height=HEIGHT):
    # The image component returns displayed pixels, which vary by screen size.
    px = px * WIDTH / display_width
    py = py * HEIGHT / display_height
    x = 60.0 + (px - LEFT) / (RIGHT - LEFT) * 60.0
    y = (py - TOP) / (BOTTOM - TOP) * 80.0
    return int(round(max(60, min(120, x)))), int(round(max(0, min(80, y))))

pitch = Image.new("RGB", (WIDTH, HEIGHT), "#155c3a")
draw = ImageDraw.Draw(pitch)

# Alternating grass stripes.
for i in range(6):
    if i % 2 == 1:
        stripe_x0 = LEFT + i * (RIGHT - LEFT) // 6
        stripe_x1 = LEFT + (i + 1) * (RIGHT - LEFT) // 6
        draw.rectangle((stripe_x0, TOP, stripe_x1, BOTTOM), fill="#196540")

def outline(x1, y1, x2, y2, width=3):
    a, b = pitch_to_pixels(x1, y1), pitch_to_pixels(x2, y2)
    draw.rectangle((*a, *b), outline="white", width=width)

outline(60, 0, 120, 80)
outline(102, 18, 120, 62)
outline(114, 30, 120, 50)

# Goalposts on right side.
gx, gy1 = pitch_to_pixels(120, 36)
_, gy2 = pitch_to_pixels(120, 44)
draw.line((gx, gy1, gx, gy2), fill="#ffdf33", width=7)

# Penalty mark.
px, py = pitch_to_pixels(108, 40)
draw.ellipse((px-4, py-4, px+4, py+4), fill="white")

# Current selected shot marker.
display_x = 108 if is_penalty else st.session_state.pitchiq_shot_x
display_y = 40 if is_penalty else st.session_state.pitchiq_shot_y
sx, sy = pitch_to_pixels(display_x, display_y)
draw.ellipse((sx-12, sy-12, sx+12, sy+12), fill="#ffad1f", outline="white", width=3)

if is_penalty:
    st.image(pitch, caption="Penalty spot (fixed)", width=840)
    clicked = None
else:
    with st.container(width=840):
        clicked = streamlit_image_coordinates(
            pitch,
            key=f"pitchiq_click_anywhere_{st.session_state.pitchiq_pitch_version}",
            width="stretch",
            cursor="crosshair",
        )

# Coordinate click components return the most recent click on reruns.
# Track it so moving another widget doesn't override a slider change.
if "pitchiq_last_click" not in st.session_state:
    st.session_state.pitchiq_last_click = None

if not is_penalty and clicked is not None and "x" in clicked and "y" in clicked:
    click_id = (clicked.get("unix_time"), int(clicked["x"]), int(clicked["y"]))
    if click_id != st.session_state.pitchiq_last_click:
        st.session_state.pitchiq_last_click = click_id
        chosen_x, chosen_y = pixels_to_pitch(
            clicked["x"], clicked["y"], clicked.get("width") or WIDTH, clicked.get("height") or HEIGHT,
        )
        st.session_state.pitchiq_shot_x = min(118, chosen_x)
        st.session_state.pitchiq_shot_y = chosen_y
        st.rerun()

st.caption("Coordinate sliders are disabled for penalties." if is_penalty else "You can also use the sliders for precise coordinates.")
slide_left, slide_right = st.columns(2)
with slide_left:
    st.slider("Shot X (closer to goal →)", min_value=60, max_value=118, step=1, key="pitchiq_shot_x", disabled=is_penalty)
with slide_right:
    st.slider("Shot Y (goal center = 40)", min_value=0, max_value=80, step=1, key="pitchiq_shot_y", disabled=is_penalty)

x = 108 if is_penalty else st.session_state.pitchiq_shot_x
y = 40 if is_penalty else st.session_state.pitchiq_shot_y
prediction = shot_prediction(model_info, player_shots, selected_goalkeeper, x, y, body_part, shot_type)
distance, angle = prediction["distance"], prediction["angle"]
standard_probability = prediction["standard"]
player_probability = prediction["player_probability"]
probability = prediction["probability"]
profile, foot_profile = prediction["profile"], prediction["foot_profile"]
multiplier, keeper_multiplier = prediction["multiplier"], prediction["keeper_multiplier"]


st.subheader("5. Predicted scoring probability")
m1, m2, m3 = st.columns(3)
m1.metric("Shooter + goalkeeper xG", f"{probability:.3f}")
m2.metric("Chance of a goal", f"{probability:.2%}")
m3.metric("Shooter-only xG", f"{player_probability:.3f}")
st.caption(f"Standard xG: {standard_probability:.3f} • Goalkeeper odds factor: ×{keeper_multiplier:.3f}" + (" (neutral for penalties)" if is_penalty else ""))
st.progress(max(0.0, min(1.0, probability)))
st.caption(f"Distance: {distance:.1f} StatsBomb pitch units • Angle: {angle:.1f}°")
if profile:
    st.caption(
        f"Player-specific odds adjustment: ×{multiplier:.3f}, estimated from "
        f"{foot_profile['shots'] if foot_profile else profile['shots']:,} training shots "
        f"({body_part} profile). This is a conservative historical "
        "finishing adjustment, not a reliable estimate of current skill."
    )
else:
    st.caption(
        "No reliable training sample for this shooter (or model not yet retrained). "
        "Player-adjusted xG equals standard xG."
    )
st.info("Shooter and goalkeeper adjustments are experimental historical factors, not a direct physical simulation or proof of current ability. Keeper factors use non-penalty shots only, so selecting a keeper will not affect penalty predictions. Some keepers may have a neutral factor.")

with st.expander("Model information"):
    st.write(f"Training shots: {model_info['training_shots']:,}")
    st.write(f"Testing shots: {model_info['testing_shots']:,}")
    st.write(f"{'Synthetic holdout' if using_demo() else 'Held-out match'} Brier score: {model_info['brier_score']:.4f}")
    st.write("Input features: distance, shot angle, body part, and shot type.")
    if "player_brier_score" in model_info:
        st.write(f"Player-adjusted held-out Brier score: {model_info['player_brier_score']:.4f}")
        st.write("Player adjustments were computed from training-match goals and reference xG, not test outcomes.")
    if "combined_brier_score" in model_info:
        st.write(f"Shooter + goalkeeper held-out Brier score: {model_info['combined_brier_score']:.5f}")
        st.write("Goalkeeper profiles use only identifiable opposing keepers, and are estimated using training-match non-penalty goals vs reference xG.")

footer("Shot data: StatsBomb Open Data · Experimental historical shooter and keeper adjustments.")
