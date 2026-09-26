import unittest

import numpy as np

from sports.handball.config import HandballCourtConfiguration
from sports.handball.roster import (
    GOALKEEPER,
    FIELD_PLAYER,
    REFEREE,
    HandballRoster,
    JerseyColorTeams,
    cluster_two_teams,
    distance_to_segments,
    substitution_segments,
)


CONFIG = HandballCourtConfiguration()


def _box(index: int) -> np.ndarray:
    x = 40.0 + index * 50.0
    return np.array([x, 80.0, x + 24.0, 180.0], dtype=np.float64)


def _feed(roster, feet, teams, kinds, scores=None, boxes=None):
    count = len(feet)
    if scores is None:
        scores = np.full(count, 0.8, dtype=np.float64)
    if boxes is None:
        if count == 0:
            boxes = np.zeros((0, 4), dtype=np.float64)
        else:
            boxes = np.stack([_box(index) for index in range(count)])
    return roster.update(
        xyxy=boxes,
        confidence=np.asarray(scores, dtype=np.float64),
        kind=kinds,
        team_id=np.asarray(teams, dtype=int),
        foot_court=np.asarray(feet, dtype=np.float64),
    )


class SubstitutionGeometryTest(unittest.TestCase):
    def test_segments_cover_both_sidelines_around_the_centre(self):
        segments = substitution_segments(CONFIG)
        self.assertEqual(segments.shape, (2, 2, 2))
        centre_touch = np.array([[2000.0, 0.0], [2000.0, 2000.0]])
        distance = distance_to_segments(centre_touch, segments)
        self.assertTrue(np.all(distance < 1.0))
        goal_mouth = np.array([[200.0, 0.0]])
        self.assertGreater(float(distance_to_segments(goal_mouth, segments)[0]), 1000.0)


class HandballRosterTest(unittest.TestCase):
    def _roster(self, **kwargs):
        defaults = dict(confirm_hits=3, warmup_frames=4, exit_hits=3)
        defaults.update(kwargs)
        return HandballRoster(CONFIG, **defaults)

    def test_a_player_inside_the_court_stays_when_the_detector_misses(self):
        roster = self._roster()
        foot = [[1800.0, 900.0]]
        for _ in range(3):
            tracks = _feed(roster, foot, [0], [FIELD_PLAYER])
        self.assertEqual(len(tracks), 1)
        self.assertTrue(tracks[0].confirmed)

        for _ in range(12):
            tracks = _feed(roster, [], [], [])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)
        self.assertGreater(tracks[0].missed, 0)
        self.assertLess(abs(tracks[0].foot_court[0] - 1800.0), 50.0)

    def test_a_one_frame_false_detection_does_not_stick(self):
        roster = self._roster()
        _feed(roster, [[1500.0, 800.0]], [0], [FIELD_PLAYER])
        for _ in range(3):
            tracks = _feed(roster, [], [], [])
        self.assertEqual(tracks, [])

    def test_each_team_keeps_six_field_players_and_one_goalkeeper(self):
        roster = self._roster()
        feet = [[400.0 + index * 300.0, 700.0] for index in range(7)]
        feet.append([2200.0, 400.0])  # goalkeeper
        teams = [0] * 8
        kinds = [FIELD_PLAYER] * 7 + [GOALKEEPER]
        scores = [0.95 - index * 0.05 for index in range(7)] + [0.9]
        tracks = _feed(roster, feet, teams, kinds, scores=scores)
        field = [track for track in tracks if track.kind == FIELD_PLAYER]
        goalkeepers = [track for track in tracks if track.kind == GOALKEEPER]
        self.assertEqual(len(field), 6)
        self.assertEqual(len(goalkeepers), 1)
        self.assertNotIn(0.65, [round(track.confidence, 2) for track in field])

        other = [[500.0 + index * 280.0, 1400.0] for index in range(6)]
        tracks = _feed(roster, other, [1] * 6, [FIELD_PLAYER] * 6)
        self.assertEqual(len([track for track in tracks if track.team_id == 1]), 6)
        self.assertEqual(len([track for track in tracks if track.team_id == 0 and track.kind == FIELD_PLAYER]), 6)

    def test_referees_do_not_take_a_player_slot(self):
        roster = self._roster()
        feet = [[450.0 + index * 320.0, 600.0] for index in range(6)]
        feet.extend([[800.0, 1500.0], [1600.0, 1500.0]])
        teams = [0] * 6 + [-1, -1]
        kinds = [FIELD_PLAYER] * 6 + [REFEREE, REFEREE]
        tracks = _feed(roster, feet, teams, kinds)
        self.assertEqual(len([track for track in tracks if track.kind == FIELD_PLAYER]), 6)
        self.assertEqual(len([track for track in tracks if track.kind == REFEREE]), 2)

    def test_after_warmup_new_players_enter_only_through_the_substitution_line(self):
        roster = self._roster(warmup_frames=2, confirm_hits=2)
        _feed(roster, [[1000.0, 1000.0]], [0], [FIELD_PLAYER])
        _feed(roster, [[1000.0, 1000.0]], [0], [FIELD_PLAYER])
        self.assertGreaterEqual(roster.frame_index, 2)

        centre = _feed(roster, [[2000.0, 1000.0]], [1], [FIELD_PLAYER], scores=[0.99])
        self.assertFalse(any(track.team_id == 1 for track in centre))

        goal_line = _feed(roster, [[150.0, 40.0]], [1], [FIELD_PLAYER], scores=[0.99])
        self.assertFalse(any(track.team_id == 1 for track in goal_line))

        entered = _feed(roster, [[2000.0, 60.0]], [1], [FIELD_PLAYER], scores=[0.99])
        newcomers = [track for track in entered if track.team_id == 1]
        self.assertEqual(len(newcomers), 1)
        self.assertLess(abs(newcomers[0].foot_court[0] - 2000.0), 1.0)

    def test_a_player_leaves_only_through_the_substitution_line(self):
        roster = self._roster(warmup_frames=1, confirm_hits=2, exit_hits=3)
        for _ in range(2):
            _feed(roster, [[300.0, 800.0], [2000.0, 500.0]], [0, 1], [FIELD_PLAYER, FIELD_PLAYER])

        foot_y = 800.0
        for _ in range(12):
            foot_y -= 90.0
            tracks = _feed(roster, [[300.0, foot_y]], [0], [FIELD_PLAYER])
        stayed = [track for track in tracks if track.team_id == 0]
        self.assertEqual(len(stayed), 1)
        self.assertGreater(stayed[0].foot_court[1], -50.0)

        foot_y = 500.0
        tracks = []
        for _ in range(8):
            foot_y -= 120.0
            tracks = _feed(roster, [[2000.0, foot_y]], [1], [FIELD_PLAYER])
            if not any(track.team_id == 1 for track in tracks):
                break
        self.assertFalse(any(track.team_id == 1 for track in tracks))
        self.assertTrue(any(track.team_id == 0 for track in tracks))

    def test_a_full_team_rejects_another_substitution(self):
        roster = self._roster(warmup_frames=1, confirm_hits=1)
        feet = [[400.0 + index * 250.0, 900.0] for index in range(6)]
        _feed(roster, feet, [0] * 6, [FIELD_PLAYER] * 6)
        tracks = _feed(roster, [[2000.0, 50.0]], [0], [FIELD_PLAYER], scores=[0.99])
        self.assertEqual(len([track for track in tracks if track.kind == FIELD_PLAYER]), 6)


class JerseyColorTest(unittest.TestCase):
    def test_dark_and_light_kits_stay_in_a_fixed_order(self):
        dark = np.array([[20.0, 20.0, 30.0]] * 5)
        light = np.array([[210.0, 210.0, 220.0]] * 5)
        colors = np.vstack([light, dark])
        labels, centroids = cluster_two_teams(colors)
        self.assertTrue(np.all(labels[5:] == 0))
        self.assertTrue(np.all(labels[:5] == 1))
        self.assertLess(centroids[0].mean(), centroids[1].mean())

        teams = JerseyColorTeams(min_samples=6, min_luma_gap=18.0)
        teams.observe(colors)
        assigned = teams.assign(np.array([[15.0, 15.0, 20.0], [200.0, 200.0, 210.0]]))
        self.assertEqual(assigned.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
