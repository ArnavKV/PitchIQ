"""Every portfolio tool must work without third-party data or artifacts."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from pitchiq.data import load_matches, load_model, load_player_frames, load_shots, using_demo
from pitchiq.demo import CUP_COMPETITION, DOMESTIC_COMPETITION, DOMESTIC_TEAMS, PLAYERS

ROOT = Path(__file__).resolve().parent.parent


class DemoWebsiteTests(unittest.TestCase):
    def setUp(self):
        mode = patch.dict(os.environ, {"PITCHIQ_DATA_MODE": "demo"})
        mode.start()
        self.addCleanup(mode.stop)

    def app(self, filename=None):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
        if filename:
            app.switch_page(f"pages/{filename}").run()
        self.healthy(app)
        return app

    def healthy(self, app):
        self.assertFalse(app.exception, [item.message for item in app.exception])
        self.assertFalse(app.error, [item.value for item in app.error])

    def test_demo_never_reads_provider_files(self):
        with patch("pandas.read_csv", side_effect=AssertionError("Provider file read")), patch("joblib.load", side_effect=AssertionError("Binary artifact read")):
            self.assertTrue(using_demo())
            self.assertTrue(load_matches().Synthetic.all())
            self.assertTrue(load_shots().Synthetic.all())
            self.assertEqual(len(load_player_frames()), 2)
            self.assertTrue(load_model("xg_predictor.joblib")["synthetic"])

    def test_missing_assets_select_one_synthetic_backend(self):
        with patch.dict(os.environ, {"PITCHIQ_DATA_MODE": "auto"}), patch("pitchiq.data.MATCHES_PATH", ROOT / "data/no-such-demo-fixture.csv"):
            self.assertTrue(using_demo())
            self.assertTrue(load_matches().Synthetic.all())
            self.assertTrue(load_model("match_predictor_v2.joblib")["synthetic"])

    def test_all_pages_load_and_disclose_fictional_data(self):
        app = self.app()
        self.assertEqual(app.metric[0].label, "Demo matches")
        for page in sorted((ROOT / "pages").glob("*.py")):
            with self.subTest(page=page.name):
                app.switch_page(f"pages/{page.name}").run()
                self.healthy(app)
                self.assertTrue(any("Fictional demo" in notice.value for notice in app.info))

    def test_player_comparison_and_search(self):
        app = self.app("1_Player_Comparison.py")
        self.assertNotEqual(app.selectbox[2].value, app.selectbox[3].value)
        self.assertTrue(any(PLAYERS[0] in title.value for title in app.subheader))
        app.text_input[0].set_value("Demo Striker 12").run()
        self.healthy(app)
        self.assertTrue(any(PLAYERS[-1] in title.value for title in app.subheader))
        self.assertTrue(app.dataframe[0].value["Source"].eq("Synthetic sample").all())

    def test_team_filter_has_complete_fictional_season(self):
        app = self.app("2_Match_Explorer.py")
        source = next(widget for widget in app.radio if widget.label == "Match data source")
        if source.value != "Fictional demo":
            source.set_value("Fictional demo").run()
        next(widget for widget in app.selectbox if widget.label == "Competition").select(DOMESTIC_COMPETITION).run()
        next(widget for widget in app.selectbox if widget.label == "Team").select(DOMESTIC_TEAMS[0]).run()
        self.healthy(app)
        self.assertEqual(app.metric[0].value, "38")
        next(widget for widget in app.selectbox if widget.label in ("Season", "Season filter")).select("2020-21").run()
        self.healthy(app)
        self.assertEqual(app.metric[0].value, "38")


    def test_penalties_keep_goalkeeper_neutral(self):
        app = self.app("4_xG_Predictor.py")
        app.selectbox[3].select("Penalty").run()
        self.healthy(app)
        probability = next(metric.value for metric in app.metric if metric.label == "Chance of a goal")
        self.assertTrue(all(slider.disabled for slider in app.slider))
        app.selectbox[1].select("Average / unspecified goalkeeper").run()
        self.healthy(app)
        self.assertEqual(probability, next(metric.value for metric in app.metric if metric.label == "Chance of a goal"))

    def test_fictional_league_and_cup_simulations_run(self):
        app = self.app("6_League_Simulator.py")
        competition = next(w for w in app.selectbox if w.label == "Competition")
        self.assertEqual(competition.value, DOMESTIC_COMPETITION)
        next(w for w in app.slider if w.label == "League simulations").set_value(200).run()
        next(w for w in app.button if "Simulate remaining" in w.label).click().run()
        self.healthy(app)
        self.assertEqual(len(app.session_state["league_2026_run"][1]), 20)
        competition = next(w for w in app.selectbox if w.label == "Competition")
        competition.select(CUP_COMPETITION).run()
        next(w for w in app.slider if w.label == "Cup simulations").set_value(200).run()
        next(w for w in app.button if "Simulate knockout" in w.label).click().run()
        self.healthy(app)
        self.assertEqual(len(app.session_state["ucl_2026_run"][1]), 16)


if __name__ == "__main__":
    unittest.main()
