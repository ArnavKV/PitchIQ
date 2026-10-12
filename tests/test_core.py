"""Regressions for data integrity, shared scoring, and historical simulations."""
import unittest
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from pitchiq.data import validate_matches
from pitchiq.predictions import scoreline_distribution
from pitchiq.shots import adjust_probability, shot_geometry, shot_prediction
from pitchiq.simulation import current_table, league_simulate, snapshot, ucl_simulate


class GoalModel:
    """Deliberately reverse class order to detect hard-coded column selection."""
    classes_ = np.array([1, 0])

    def predict_proba(self, frame):
        return np.array([[0.2, 0.8]])


class ConstantGoals:
    def predict(self, frame):
        return np.full(len(frame), 1.3)


def fixtures():
    return pd.DataFrame([
        {"Date": "2023-01-01", "Season": "2022-23", "Competition": "Test league", "HomeTeam": "A", "AwayTeam": "B", "HomeGoals": 2, "AwayGoals": 0, "Result": "H"},
        {"Date": "2023-01-10", "Season": "2022-23", "Competition": "Test league", "HomeTeam": "B", "AwayTeam": "A", "HomeGoals": 1, "AwayGoals": 1, "Result": "D"},
    ])


class DataTests(unittest.TestCase):
    def test_invalid_and_duplicate_matches_are_removed(self):
        frame = fixtures()
        duplicate = frame.iloc[[0]]
        invalid = frame.iloc[[0]].assign(Date="not a date")
        fractional = frame.iloc[[0]].assign(HomeGoals=1.5)
        negative = frame.iloc[[0]].assign(AwayGoals=-1)
        clean = validate_matches(pd.concat([frame, duplicate, invalid, fractional, negative]))
        self.assertEqual(len(clean), 2)
        self.assertTrue(pd.api.types.is_integer_dtype(clean.HomeGoals))

    def test_missing_season_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Season"):
            validate_matches(fixtures().drop(columns="Season"))


class ShotTests(unittest.TestCase):
    def setUp(self):
        self.info = {
            "model": GoalModel(), "features": ["ShotDistance", "ShotAngle", "BodyPart", "ShotType"],
            "player_factors": {"10": {"multiplier": 1.2}},
            "player_foot_factors": {"10|Right Foot": {"multiplier": 1.5}},
            "goalkeeper_factors": {"Keeper": {"multiplier": 0.8}},
        }
        self.shots = pd.DataFrame({"ShooterID": [10]})

    def test_centered_penalty_geometry(self):
        distance, angle = shot_geometry(108, 40)
        self.assertEqual(distance, 12)
        self.assertGreater(angle, 0)
        self.assertEqual(shot_geometry(102, 30), shot_geometry(102, 50))

    def test_adjustment_preserves_boundaries(self):
        self.assertEqual(adjust_probability(0, 2), 0)
        self.assertEqual(adjust_probability(1, 2), 1)
        self.assertAlmostEqual(adjust_probability(0.2, 1), 0.2)
        with self.assertRaises(ValueError):
            adjust_probability(0.2, -1)

    def test_scoring_class_and_foot_profile_are_respected(self):
        result = shot_prediction(self.info, self.shots, "Keeper", 102, 40, "Right Foot", "Open Play")
        self.assertEqual(result["standard"], 0.2)
        self.assertAlmostEqual(result["probability"], adjust_probability(0.2, 1.5 * 0.8))

    def test_penalty_ignores_goalkeeper_factor(self):
        keeper = shot_prediction(self.info, self.shots, "Keeper", 108, 40, "Right Foot", "Penalty")
        neutral = shot_prediction(self.info, self.shots, "Unspecified", 108, 40, "Right Foot", "Penalty")
        self.assertEqual(keeper["probability"], neutral["probability"])

    def test_absent_player_profile_uses_standard_xg(self):
        result = shot_prediction(self.info, pd.DataFrame(), "Unspecified", 102, 40, "Right Foot", "Open Play")
        self.assertAlmostEqual(result["probability"], result["standard"])


class SimulationTests(unittest.TestCase):
    def test_cutoff_excludes_same_day_and_future_scores(self):
        frame = validate_matches(fixtures())
        history, _, last = snapshot(frame, "Test league", pd.Timestamp("2023-01-10"))
        self.assertEqual(len(history["A"]), 1)
        self.assertEqual(history["A"][0]["gf"], 2)
        self.assertEqual(last["A"], pd.Timestamp("2023-01-01"))
        other = frame.assign(Competition="Other league", HomeGoals=99)
        with_other = snapshot(pd.concat([frame, other]), "Test league", pd.Timestamp("2023-01-10"))
        self.assertEqual(history, with_other[0])

    def test_completed_season_has_exact_points(self):
        played = validate_matches(fixtures())
        standings = current_table(played, ["A", "B"])
        self.assertEqual(standings.Pts.tolist(), [4, 1])
        report = league_simulate(played, played.iloc[:0], ["A", "B"], np.empty((0, 2)), 200, 42)
        self.assertEqual(report["Projected pts"].tolist(), [4.0, 1.0])
        self.assertEqual(report["Title %"].tolist(), [100.0, 0.0])

    def test_seed_repeats_league_report(self):
        frame = validate_matches(fixtures())
        args = (frame.iloc[:0], frame, ["A", "B"], np.array([[1.5, 0.8], [0.7, 1.6]]), 400, 42)
        pd.testing.assert_frame_equal(league_simulate(*args), league_simulate(*args))

    def test_cup_repeats_bracket_and_results(self):
        entrants = [f"Team {number:02}" for number in range(16)]
        fields = [f"{prefix}_{metric}" for prefix in ("HomeRecent", "AwayRecent", "HomeAtHome", "AwayAtAway") for metric in ("Points", "Scored", "Conceded", "WinRate")]
        fields += ["EloDifference", "HomeElo", "AwayElo", "HomeDaysRest", "AwayDaysRest"]
        model = {"features": fields, "poisson_home_model": ConstantGoals(), "poisson_away_model": ConstantGoals()}
        args = (entrants, pd.Timestamp("2023-01-01"), {}, {}, {}, model, 400, 42)
        first, bracket = ucl_simulate(*args)
        second, other_bracket = ucl_simulate(*args)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(bracket, other_bracket)
        self.assertAlmostEqual(first["Win trophy %"].sum(), 100, delta=1)
        self.assertEqual([len(ties) for _, ties in bracket], [8, 4, 2, 1])

    def test_cup_seed_survives_a_python_restart(self):
        # A set's iteration order changes between processes; a seed alone is
        # insufficient unless participant-pair sampling also has stable order.
        script = '''
import json, runpy
d = runpy.run_path("tests/test_core.py")
pd = d["pd"]
teams = [f"Team {n:02}" for n in range(16)]
fields = [f"{p}_{m}" for p in ("HomeRecent", "AwayRecent", "HomeAtHome", "AwayAtAway") for m in ("Points", "Scored", "Conceded", "WinRate")]
fields += ["EloDifference", "HomeElo", "AwayElo", "HomeDaysRest", "AwayDaysRest"]
model = {"features": fields, "poisson_home_model": d["ConstantGoals"](), "poisson_away_model": d["ConstantGoals"]()}
report, bracket = d["ucl_simulate"](teams, pd.Timestamp("2023-01-01"), {}, {}, {}, model, 200, 42)
print(json.dumps({"report": report.to_dict(), "bracket": bracket}, sort_keys=True))
'''
        outputs = []
        for hash_seed in (1, 99):
            result = subprocess.run(
                [sys.executable, "-c", script], cwd=Path(__file__).resolve().parent.parent,
                env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
                check=True, text=True, capture_output=True,
            )
            outputs.append(json.loads(result.stdout))
        self.assertEqual(outputs[0], outputs[1])

    def test_scorelines_are_sorted_probabilities(self):
        scores = scoreline_distribution(1.5, 1.1)
        probabilities = [probability for _, _, probability in scores]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        self.assertTrue(all(0 <= p <= 1 for p in probabilities))
        self.assertLessEqual(sum(probabilities), 1)


if __name__ == "__main__":
    unittest.main()
