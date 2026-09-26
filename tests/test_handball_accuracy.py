import unittest

import cv2
import numpy as np
import supervision as sv

from sports.common.ball import BallTracker
from sports.handball.annotators import free_throw_polyline, split_dashes
from sports.handball.ball import HandballBallTracker
from sports.handball.calibration import (
    CourtCalibrator,
    court_correspondences,
    estimate_court_projection,
    inside_court,
)
from sports.handball.config import HandballCourtConfiguration


def _polyline_length(polyline: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())


def _camera_homography() -> np.ndarray:
    config = HandballCourtConfiguration()
    court_corners = np.array([
        config.vertices[0],
        config.vertices[2],
        config.vertices[3],
        config.vertices[5],
    ], dtype=np.float32)
    image_corners = np.array([
        [140, 90],
        [980, 160],
        [920, 640],
        [70, 600],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(court_corners, image_corners)


def _project_court(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    shaped = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(shaped, matrix).reshape(-1, 2)


def _court_error_cm(image_to_court: np.ndarray, image_xy: np.ndarray, court_xy: np.ndarray) -> float:
    shaped = np.asarray(image_xy, dtype=np.float32).reshape(-1, 1, 2)
    recovered = cv2.perspectiveTransform(shaped, image_to_court).reshape(-1, 2)
    return float(np.linalg.norm(recovered - court_xy, axis=1).mean())


class FreeThrowMarkingTest(unittest.TestCase):
    def test_dashes_are_15cm_along_the_whole_line(self):
        config = HandballCourtConfiguration()
        for side in ("left", "right"):
            polyline = free_throw_polyline(config, side)
            dashes = split_dashes(
                polyline,
                config.free_throw_line_segment_length,
                config.free_throw_line_gap_length,
            )
            full_dashes = dashes[:-1]
            lengths = [_polyline_length(dash) for dash in full_dashes]
            self.assertTrue(lengths)
            self.assertTrue(all(abs(length - 15) < 0.05 for length in lengths))
            self.assertLessEqual(_polyline_length(dashes[-1]), 15.05)

            covered = 0.0
            for dash in dashes[:-1]:
                covered += _polyline_length(dash) + config.free_throw_line_gap_length
            # The gap after every full dash is 15 cm, including the gap before a short final dash.
            self.assertAlmostEqual(
                covered + _polyline_length(dashes[-1]),
                _polyline_length(polyline),
                delta=0.05,
            )

    def test_free_throw_line_stays_on_the_9m_construction(self):
        config = HandballCourtConfiguration()
        polyline = free_throw_polyline(config, "left")
        top = np.array([0.0, config.goal_top_y])
        bottom = np.array([0.0, config.goal_bottom_y])
        for point in polyline:
            on_arc = (
                abs(np.linalg.norm(point - top) - config.free_throw_radius) < 1e-4
                or abs(np.linalg.norm(point - bottom) - config.free_throw_radius) < 1e-4
            )
            on_straight = (
                abs(point[0] - config.free_throw_radius) < 1e-4
                and config.goal_top_y - 1e-4 <= point[1] <= config.goal_bottom_y + 1e-4
            )
            self.assertTrue(on_arc or on_straight)
        self.assertAlmostEqual(polyline[0, 1], 0.0, places=4)
        self.assertAlmostEqual(polyline[-1, 1], config.width, places=4)


class CourtCalibrationTest(unittest.TestCase):
    def test_confidence_gate_keeps_a_real_corner_on_the_origin(self):
        image = np.array([[10, 10], [20, 20], [30, 30], [0, 0]], dtype=np.float32)
        court = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)
        confidence = np.array([0.9, 0.2, 0.8, 0.99], dtype=np.float32)

        kept_image, kept_court = court_correspondences(image, court, confidence, 0.35)
        self.assertEqual(len(kept_image), 3)
        self.assertTrue(np.allclose(kept_court[:, 0], [1, 3, 4]))

        kept_without_scores, _ = court_correspondences(image, court, None)
        self.assertEqual(len(kept_without_scores), 3)
        self.assertFalse(np.any(np.all(kept_without_scores == 0, axis=1)))
    def _observed_keypoints(self, noise_px: float, outlier_count: int, seed: int):
        config = HandballCourtConfiguration()
        court = np.array(config.vertices, dtype=np.float32)
        court_to_image = _camera_homography()
        image = _project_court(court_to_image, court)
        rng = np.random.default_rng(seed)
        noisy = image + rng.normal(0, noise_px, image.shape)
        if outlier_count:
            indices = rng.choice(len(noisy), size=outlier_count, replace=False)
            noisy[indices] = rng.uniform([0, 0], [1100, 720], size=(outlier_count, 2))
        confidence = np.full(len(noisy), 0.9, dtype=np.float32)
        outlier_mask = np.zeros(len(noisy), dtype=bool)
        if outlier_count:
            confidence[indices] = 0.99
            outlier_mask[indices] = True
        return court, noisy.astype(np.float32), confidence, outlier_mask

    def test_outliers_do_not_move_a_player_by_metres(self):
        court, noisy, confidence, _ = self._observed_keypoints(
            noise_px=1.2,
            outlier_count=8,
            seed=7,
        )
        robust = estimate_court_projection(noisy, court, confidence)
        self.assertIsNotNone(robust)

        naive, _ = cv2.findHomography(noisy, court, 0)
        player_court = np.array([[2100, 1000]], dtype=np.float32)
        court_to_image = _camera_homography()
        player_image = _project_court(court_to_image, player_court)

        robust_error = _court_error_cm(robust.image_to_court, player_image, player_court)
        naive_error = _court_error_cm(naive, player_image, player_court)
        self.assertLess(robust_error, 15.0)
        self.assertGreater(naive_error, 100.0)

    def test_low_confidence_keypoints_are_ignored(self):
        court, noisy, confidence, outliers = self._observed_keypoints(
            noise_px=0.4,
            outlier_count=6,
            seed=3,
        )
        confidence[outliers] = 0.05
        projection = estimate_court_projection(noisy, court, confidence)
        self.assertIsNotNone(projection)
        self.assertGreaterEqual(projection.inlier_count, len(court) - 6)

    def test_static_camera_jitter_is_smoothed(self):
        config = HandballCourtConfiguration()
        court = np.array(config.vertices, dtype=np.float32)
        court_to_image = _camera_homography()
        image = _project_court(court_to_image, court)
        player_image = _project_court(court_to_image, np.array([[1800, 700]], np.float32))
        rng = np.random.default_rng(11)
        raw = CourtCalibrator(court, alpha=1.0)
        smooth = CourtCalibrator(court, alpha=0.35)
        raw_positions = []
        smooth_positions = []
        for _ in range(25):
            observed = image + rng.normal(0, 1.5, image.shape)
            confidence = np.full(len(court), 0.9, dtype=np.float32)
            raw_projection = raw.update(observed.astype(np.float32), confidence)
            smooth_projection = smooth.update(observed.astype(np.float32), confidence)
            raw_positions.append(raw_projection.transform_points(player_image)[0])
            smooth_positions.append(smooth_projection.transform_points(player_image)[0])

        raw_std = float(np.std(raw_positions, axis=0).mean())
        smooth_std = float(np.std(smooth_positions, axis=0).mean())
        self.assertLess(smooth_std, raw_std * 0.6)

    def test_failed_frames_keep_the_last_good_calibration(self):
        court, noisy, confidence, _ = self._observed_keypoints(0.2, 0, seed=1)
        calibrator = CourtCalibrator(court, max_hold_frames=2)
        self.assertIsNotNone(calibrator.update(noisy, confidence))
        empty = np.zeros((0, 2), dtype=np.float32)
        self.assertIsNotNone(calibrator.update(empty))
        self.assertIsNotNone(calibrator.update(empty))
        self.assertIsNone(calibrator.update(empty))

    def test_points_outside_the_safety_zone_are_rejected(self):
        points = np.array([
            [2000, 1000],
            [-150, 1000],
            [-250, 1000],
            [4300, 1000],
        ], dtype=np.float32)
        mask = inside_court(points, length_cm=4000, width_cm=2000, margin_cm=200)
        self.assertTrue(mask[0] and mask[1])
        self.assertFalse(mask[2] or mask[3])


class HandballBallTrackerTest(unittest.TestCase):
    def test_fast_pass_ignores_a_static_false_detection(self):
        tracker = HandballBallTracker()
        centroid = _CentroidTracker()
        errors = []
        centroid_errors = []
        for index in range(20):
            ball = np.array([20.0 + index * 75.0, 180.0])
            false_positive = np.array([420.0, 420.0])
            measurements = np.vstack([ball, false_positive])
            confidences = np.array([0.82, 0.55])
            estimated = tracker.update(measurements, confidences)
            centroid_pick = centroid.update(measurements)
            errors.append(np.linalg.norm(estimated - ball))
            centroid_errors.append(np.linalg.norm(centroid_pick - ball))

        self.assertLess(np.median(errors[3:]), 8.0)
        self.assertGreater(np.median(centroid_errors[8:]), 40.0)

    def test_shot_is_confirmed_on_the_next_frame(self):
        tracker = HandballBallTracker()
        positions = [np.array([100.0 + index * 6.0, 240.0]) for index in range(8)]
        shot_origin = positions[-1] + np.array([96.0, 0.0])
        positions.extend(shot_origin + np.array([90.0, 0.0]) * step for step in range(4))

        errors = []
        for position in positions:
            estimated = tracker.update(position.reshape(1, 2), np.array([0.8]))
            errors.append(np.linalg.norm(estimated - position))

        # The shot frame coasts on the old velocity. The following frames lock on.
        self.assertLess(np.max(errors[9:]), 12.0)

    def test_a_false_detection_during_a_gap_does_not_steal_the_track(self):
        tracker = HandballBallTracker()
        for index in range(6):
            ball = np.array([50.0 + index * 18.0, 300.0])
            tracker.update(ball.reshape(1, 2), np.array([0.85]))

        next_ball = np.array([50.0 + 6 * 18.0, 300.0])
        false_positive = next_ball + np.array([110.0, -80.0])
        coasted = tracker.update(false_positive.reshape(1, 2), np.array([0.95]))
        self.assertLess(np.linalg.norm(coasted - next_ball), 20.0)

        resumed = next_ball + np.array([18.0, 0.0])
        recovered = tracker.update(resumed.reshape(1, 2), np.array([0.85]))
        self.assertLess(np.linalg.norm(recovered - resumed), 15.0)

    def test_centroid_tracker_selects_the_false_ball_on_the_same_pass(self):
        """The shared centroid tracker is the baseline this filter replaces."""
        tracker = BallTracker(buffer_size=10)
        false_selected = 0
        for index in range(15):
            ball = np.array([20.0 + index * 75.0, 180.0])
            false_positive = np.array([420.0, 420.0])
            detections = sv.Detections(
                xyxy=_boxes(np.vstack([ball, false_positive])),
                confidence=np.array([0.82, 0.55], dtype=np.float32),
                class_id=np.array([0, 0]),
            )
            chosen = tracker.update(detections)
            center = chosen.get_anchors_coordinates(sv.Position.CENTER)[0]
            if np.linalg.norm(center - false_positive) < np.linalg.norm(center - ball):
                false_selected += 1
        self.assertGreater(false_selected, 4)


class _CentroidTracker:
    def __init__(self):
        self.history = []

    def update(self, measurements: np.ndarray) -> np.ndarray:
        self.history.append(measurements)
        window = self.history[-10:]
        centroid = np.mean(np.concatenate(window, axis=0), axis=0)
        distances = np.linalg.norm(measurements - centroid, axis=1)
        return measurements[int(np.argmin(distances))]


def _boxes(centers: np.ndarray) -> np.ndarray:
    half = 4.0
    return np.column_stack([
        centers[:, 0] - half,
        centers[:, 1] - half,
        centers[:, 0] + half,
        centers[:, 1] + half,
    ]).astype(np.float32)


if __name__ == "__main__":
    unittest.main()
