"""Interaction smoke tests against local data and trusted trained artifacts."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest
from pitchiq.data import using_demo

ROOT = Path(__file__).resolve().parent.parent
ASSETS_READY = all((ROOT / path).exists() for path in (
    "data/matches/all_matches.csv", "data/shots/shots.csv",
    "data/player_comparison/cleaned/transfermarkt_clean.csv",
    "data/player_comparison/cleaned/understat_clean.csv",
    "models/match_predictor_v2.joblib", "models/match_predictor_v3.joblib",
    "models/xg_predictor.joblib",
))


@unittest.skipUnless(ASSETS_READY and not using_demo(), "Real-data tests require complete local assets and local/auto mode")
class WebsiteTests(unittest.TestCase):
    def app(self, page=None):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        if page:
            app.switch_page(f"pages/{page}").run()
        self.assertHealthy(app)
        return app

    def assertHealthy(self, app):
        self.assertEqual([item.message for item in app.exception], [])
        self.assertEqual([item.value for item in app.error], [])

    def test_every_navigation_page_loads(self):
        app = self.app()
        self.assertEqual(app.metric[0].label, "Recorded matches")
        for page in sorted((ROOT / "pages").glob("*.py")):
            with self.subTest(page=page.name):
                app.switch_page(f"pages/{page.name}").run()
                self.assertHealthy(app)

    def test_explorer_team_and_season_filters(self):
        app = self.app("2_Match_Explorer.py")
        app.selectbox[1].select("Premier League").run()
        app.selectbox[2].select("Arsenal FC").run()
        self.assertHealthy(app)
        self.assertEqual(app.metric[0].value, "38")
        table = app.dataframe[0].value
        self.assertTrue((table.HomeTeam.eq("Arsenal FC") | table.AwayTeam.eq("Arsenal FC")).all())
        app.selectbox[0].select("2020-21").run()
        self.assertHealthy(app)
        self.assertEqual(app.metric[0].value, "38")

    def test_player_search_and_empty_search(self):
        app = self.app("1_Player_Comparison.py")
        app.text_input[0].set_value("Haaland").run()
        self.assertHealthy(app)
        self.assertTrue(any("Erling" in title.value for title in app.subheader))
        app.text_input[0].set_value("no-such-footballer-123").run()
        self.assertHealthy(app)
        self.assertTrue(any("No matching player" in item.value for item in app.info))

    def test_penalty_location_and_keeper_neutrality(self):
        app = self.app("4_xG_Predictor.py")
        app.selectbox[3].select("Penalty").run()
        self.assertHealthy(app)
        self.assertTrue(all(slider.disabled for slider in app.slider))
        chance = next(metric.value for metric in app.metric if metric.label == "Chance of a goal")
        app.selectbox[1].select("Average / unspecified goalkeeper").run()
        self.assertHealthy(app)
        self.assertEqual(chance, next(metric.value for metric in app.metric if metric.label == "Chance of a goal"))
        app.selectbox[3].select("Open Play").run()
        self.assertHealthy(app)
        self.assertFalse(any(slider.disabled for slider in app.slider))

    def test_matchup_excludes_same_team(self):
        app = self.app("3_Match_Predictor.py")
        self.assertNotIn(app.selectbox[1].value, app.selectbox[2].options)
        app.selectbox[1].select("Arsenal FC").run()
        self.assertHealthy(app)
        self.assertNotIn("Arsenal FC", app.selectbox[2].options)

    def test_league_and_cup_run_and_invalidation(self):
        app = self.app("6_League_Simulator.py")
        app.slider[1].set_value(200).run()
        app.button[0].click().run()
        self.assertHealthy(app)
        report = app.session_state["league_run"]["report"]
        self.assertEqual(len(report), 20)
        self.assertEqual(len(app.dataframe), 2)
        app.number_input[0].set_value(43).run()
        self.assertHealthy(app)
        self.assertEqual(len(app.dataframe), 1)
        competition = next(option for option in app.selectbox[0].options if "Champions" in option)
        app.selectbox[0].select(competition).run()
        app.slider[0].set_value(200).run()
        app.button[0].click().run()
        self.assertHealthy(app)
        self.assertEqual(len(app.session_state["ucl_run"]["report"]), 16)


if __name__ == "__main__":
    unittest.main()
