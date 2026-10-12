"""Public demo contracts: original inputs, coherent fixtures, usable models."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from pitchiq import demo
from pitchiq.predictions import history_snapshots, make_feature_row, probabilities_from_model
from pitchiq.shots import shot_prediction
from pitchiq.simulation import predict_many, snapshot


class DemoDataTests(unittest.TestCase):
    def test_each_fictional_season_has_a_complete_balanced_schedule(self):
        frame = demo.matches()
        self.assertEqual(set(frame.Season), set(demo.SEASONS))
        self.assertTrue(frame.Date.is_monotonic_increasing)
        self.assertTrue(frame.Synthetic.all())
        self.assertEqual(set(frame.Source), {demo.DATA_SOURCE})
        for (competition, season), group in frame.groupby(["Competition", "Season"]):
            with self.subTest(competition=competition, season=season):
                size = 20 if competition == demo.DOMESTIC_COMPETITION else 16
                teams = set(group.HomeTeam) | set(group.AwayTeam)
                self.assertEqual(len(teams), size)
                self.assertEqual(len(group), size * (size - 1))
                self.assertTrue(all(team.startswith("Demo ") for team in teams))
                self.assertFalse(group.HomeTeam.eq(group.AwayTeam).any())
                self.assertFalse(group.duplicated(["HomeTeam", "AwayTeam"]).any())
                appearances = pd.concat([group[["Date", "HomeTeam"]].rename(columns={"HomeTeam": "Team"}), group[["Date", "AwayTeam"]].rename(columns={"AwayTeam": "Team"})])
                self.assertFalse(appearances.duplicated(["Date", "Team"]).any())
                self.assertTrue(group.HomeGoals.ge(0).all() and group.AwayGoals.ge(0).all())
                expected = np.where(group.HomeGoals > group.AwayGoals, "H", np.where(group.HomeGoals < group.AwayGoals, "A", "D"))
                np.testing.assert_array_equal(group.Result, expected)

    def test_frames_are_independent_and_player_ids_link_both_schemas(self):
        core, advanced = demo.player_frames()
        self.assertGreaterEqual(core.Player.nunique(), 8)
        self.assertEqual(set(core.ProviderPlayerID), set(advanced.ProviderPlayerID))
        self.assertEqual(set(core.Player), set(demo.shots().Shooter))
        self.assertTrue({"season", "competition_id", "Player", "ProviderPlayerID", "Games", "Goals", "Assists", "Minutes"} <= set(core))
        self.assertTrue({"Season", "league", "Player", "ProviderPlayerID", "Games", "Goals", "Assists", "Minutes", "Shots", "xG", "xA"} <= set(advanced))
        core.loc[0, "Goals"] = -999
        self.assertGreaterEqual(demo.player_frames()[0].loc[0, "Goals"], 0)
        matches = demo.matches()
        matches.loc[0, "HomeGoals"] = -999
        self.assertGreaterEqual(demo.matches().loc[0, "HomeGoals"], 0)
        shots = demo.shots()
        shots.loc[0, "Shooter"] = "mutated"
        self.assertTrue(demo.shots().loc[0, "Shooter"].startswith("Demo "))


class DemoModelTests(unittest.TestCase):
    def test_demo_generation_and_model_training_never_load_local_data(self):
        # Exercise uncached construction as well as the public cached interface.
        with patch("pandas.read_csv", side_effect=AssertionError("Read private dataset")), patch("joblib.load", side_effect=AssertionError("Loaded private model")):
            self.assertGreater(len(demo._matches.__wrapped__()), 0)
            self.assertGreater(len(demo._shots.__wrapped__()), 0)
            self.assertEqual(len(demo._player_frames.__wrapped__()), 2)
            self.assertEqual(set(demo._match_models.__wrapped__()), {"match_predictor_v2.joblib", "match_predictor_v3.joblib"})
            self.assertTrue(demo._xg_model.__wrapped__()["synthetic"])
            for filename in ("match_predictor_v2.joblib", "match_predictor_v3.joblib", "xg_predictor.joblib"):
                artifact = demo.model(filename)
                self.assertTrue(artifact["synthetic"])
                self.assertEqual(artifact["data_source"], "synthetic")

    def test_match_models_accept_existing_history_and_simulation_features(self):
        frame = demo.matches()
        outcome = demo.model("match_predictor_v2.joblib")
        goals = demo.model("match_predictor_v3.joblib")
        history, elo, last, date = history_snapshots(frame, demo.DOMESTIC_COMPETITION, "2023-24", 5, 20, 60)
        home, away = demo.DOMESTIC_TEAMS[:2]
        X = make_feature_row(home, away, history, elo, last, date + pd.Timedelta(days=7), 5, outcome["features"])
        probabilities = probabilities_from_model(outcome["model"], X)
        self.assertEqual(set(probabilities), {"H", "D", "A"})
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        self.assertTrue(all(np.isfinite(value) and 0 <= value <= 1 for value in probabilities.values()))
        cutoff = pd.Timestamp("2023-08-19")
        history, elo, last = snapshot(frame, demo.DOMESTIC_COMPETITION, cutoff)
        estimates = predict_many([(home, away, cutoff)], history, elo, last, goals)
        self.assertEqual(estimates.shape, (1, 2))
        self.assertTrue(np.isfinite(estimates).all() and (estimates > 0).all())
        # Initial fixtures intentionally lack history; the pipelines impute it.
        initial = predict_many([(home, away, cutoff)], {}, {}, {}, goals)
        self.assertTrue(np.isfinite(initial).all())
        self.assertEqual(outcome["training_matches"], 1860)
        self.assertEqual(outcome["testing_matches"], 620)

    def test_xg_model_matches_shared_predictor_and_reports_actual_generated_holdout(self):
        info = demo.model("xg_predictor.joblib")
        shots = demo.shots()
        player_shots = shots[shots.Shooter.eq(demo.PLAYERS[0])]
        result = shot_prediction(info, player_shots, demo.KEEPERS[0], 102, 40, "Right Foot", "Open Play")
        self.assertTrue(all(np.isfinite(result[field]) and 0 <= result[field] <= 1 for field in ("standard", "player_probability", "probability")))
        neutral = shot_prediction(info, player_shots, "Average / unspecified goalkeeper", 108, 40, "Right Foot", "Penalty")
        keeper = shot_prediction(info, player_shots, demo.KEEPERS[0], 108, 40, "Right Foot", "Penalty")
        self.assertEqual(neutral["probability"], keeper["probability"])
        train_ids = shots.MatchID.drop_duplicates().iloc[:96]
        test = shots[~shots.MatchID.isin(train_ids)]
        probability = info["model"].predict_proba(test[info["features"]])[:, list(info["model"].classes_).index(1)]
        self.assertAlmostEqual(info["brier_score"], brier_score_loss(test.Goal, probability))
        self.assertEqual(info["training_shots"] + info["testing_shots"], len(shots))
        self.assertEqual(info["testing_matches"], test.MatchID.nunique())
        info["features"].clear()
        self.assertEqual(demo.model("xg_predictor.joblib")["features"], list(demo.SHOT_FEATURES))

    def test_unknown_model_filename_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "No synthetic demo model"):
            demo.model("untrusted.joblib")


if __name__ == "__main__":
    unittest.main()
