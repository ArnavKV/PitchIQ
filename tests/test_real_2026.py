"""Offline regression tests for separating 2026-27 results and future fixtures."""
import json
import unittest

import pandas as pd

from pitchiq.real_matches import parse_league


class SourceResultSeparationTests(unittest.TestCase):
    def test_2026_placeholder_and_future_scores_are_not_results(self):
        rows = [
            {"date": "2026-09-08", "team1": "A", "team2": "B", "score": {"ft": [2, 1]}},
            {"date": "2026-09-09", "team1": "B", "team2": "A", "score": [0, 0]},
            {"date": "2026-10-20", "team1": "C", "team2": "A", "score": {"ft": [7, 1]}},
            {"date": "2027-02-02", "team1": "A", "team2": "C"},
        ]
        completed, remaining = parse_league(json.dumps({"matches": rows}), "2026-27", "Premier League", "2026-10-11")
        self.assertEqual(len(completed), 1)
        self.assertEqual((completed[0]["HomeGoals"], completed[0]["AwayGoals"]), (2, 1))
        self.assertEqual(len(remaining), 3)
        self.assertEqual({r["AwayTeam"] for r in remaining}, {"A", "C"})

    def test_historical_list_scores_allowed(self):
        rows = [{"date": "2025-10-01", "team1": "A", "team2": "B", "score": [1, 1]}]
        played, pending = parse_league(json.dumps({"matches": rows}), "2025-26", "Serie A", "2026-10-11")
        self.assertEqual(len(played), 1)
        self.assertEqual(played[0]["Result"], "D")
        self.assertEqual(pending, [])

    def test_2026_fixture_without_score_is_not_a_draw(self):
        rows = [{"date": "2026-10-22", "team1": "A", "team2": "B", "round": "Matchday 10"}]
        played, pending = parse_league(json.dumps({"matches": rows}), "2026-27", "La Liga", "2026-10-11")
        self.assertFalse(played)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["Round"], "Matchday 10")


if __name__ == "__main__":
    unittest.main()
