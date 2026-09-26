from typing import List, Optional, Tuple

import numpy as np
import supervision as sv


class HandballBallTracker:
    """Track one handball with a constant-velocity filter.

    ``sports.common.ball.BallTracker`` keeps the detection nearest the average
    of recent positions. A handball shot moves far enough in one frame that
    this average stays behind the ball, and a false detection closer to the
    average is selected instead.

    Smooth flight is updated with a Kalman filter inside a tight gate around
    the prediction. A shot leaves that gate. The new detection is accepted
    only when the next frame continues the same velocity, which a flickering
    false detection does not do.
    """

    def __init__(
        self,
        dt: float = 1.0,
        measurement_std: float = 3.0,
        process_accel_std: float = 18.0,
        gate_sigma: float = 3.5,
        min_gate_px: float = 16.0,
        tight_gate_px: float = 42.0,
        wide_gate_px: float = 220.0,
        confirm_px: float = 36.0,
        wide_confidence: float = 0.45,
        max_missed: int = 6,
        confidence_weight: float = 8.0,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive.")
        self.dt = float(dt)
        self.measurement_std = measurement_std
        self.process_accel_std = process_accel_std
        self.gate_sigma = gate_sigma
        self.min_gate_px = min_gate_px
        self.tight_gate_px = tight_gate_px
        self.wide_gate_px = wide_gate_px
        self.confirm_px = confirm_px
        self.wide_confidence = wide_confidence
        self.max_missed = max_missed
        self.confidence_weight = confidence_weight
        self.state: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.updates = 0
        self.missed = 0
        self.last_measurement: Optional[np.ndarray] = None
        self._pending: List[Tuple[np.ndarray, np.ndarray]] = []

    def update(
        self,
        measurements: np.ndarray,
        confidences: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Return the ball position in pixels, or None when the track is lost."""
        points = np.asarray(measurements, dtype=np.float64).reshape(-1, 2)
        if confidences is None:
            scores = np.ones(len(points), dtype=np.float64)
        else:
            scores = np.asarray(confidences, dtype=np.float64).reshape(-1)
            if len(scores) != len(points):
                raise ValueError("Confidences must match the measurements.")

        finite = np.isfinite(points).all(axis=1) & np.isfinite(scores)
        points = points[finite]
        scores = scores[finite]

        if self.state is None:
            return self._start(points, scores)

        self._predict()
        if len(points) == 0:
            return self._coast()

        distances = np.linalg.norm(points - self.state[:2], axis=1)
        chosen = self._match_tight(points, scores, distances)
        if chosen is not None:
            self._kalman_update(points[chosen])
            return self.state[:2].copy()

        confirmed = self._confirm_pending(points, scores)
        if confirmed is not None:
            return confirmed

        self._remember_wide_candidates(points, scores, distances)
        return self._coast()

    def update_detections(self, detections: sv.Detections) -> sv.Detections:
        """Return the single selected ball, or a predicted ball while coasting."""
        if len(detections) == 0:
            points = np.zeros((0, 2), dtype=np.float64)
            scores = np.zeros((0,), dtype=np.float64)
        else:
            points = detections.get_anchors_coordinates(sv.Position.CENTER)
            if detections.confidence is None:
                scores = np.ones(len(detections), dtype=np.float64)
            else:
                scores = np.asarray(detections.confidence, dtype=np.float64)

        position = self.update(points, scores)
        if position is None:
            return sv.Detections.empty()

        if len(detections) > 0:
            centers = detections.get_anchors_coordinates(sv.Position.CENTER)
            nearest = int(np.argmin(np.linalg.norm(centers - position, axis=1)))
            if np.linalg.norm(centers[nearest] - position) <= 1.0:
                return detections[[nearest]]

        half = 2.0
        x, y = float(position[0]), float(position[1])
        return sv.Detections(
            xyxy=np.array([[x - half, y - half, x + half, y + half]], dtype=np.float32),
            confidence=np.array([0.25], dtype=np.float32),
            class_id=np.array([0], dtype=int),
        )

    def _start(
        self,
        points: np.ndarray,
        scores: np.ndarray,
    ) -> Optional[np.ndarray]:
        if len(points) == 0:
            return None
        index = int(np.argmax(scores))
        self._initialize(points[index], velocity=np.zeros(2))
        return self.state[:2].copy()

    def _initialize(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.state = np.array(
            [position[0], position[1], velocity[0], velocity[1]],
            dtype=np.float64,
        )
        velocity_std = self.wide_gate_px
        self.covariance = np.diag([
            self.measurement_std ** 2,
            self.measurement_std ** 2,
            velocity_std ** 2,
            velocity_std ** 2,
        ]).astype(np.float64)
        self.last_measurement = np.asarray(position, dtype=np.float64).copy()
        self.updates = 1
        self.missed = 0
        self._pending = []

    def _predict(self) -> None:
        dt = self.dt
        transition = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        accel = self.process_accel_std ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        process = accel * np.array([
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ])
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process

    def _tight_limit(self) -> float:
        if self.updates < 2 or self.covariance is None:
            return self.wide_gate_px
        position_std = float(np.sqrt(self.covariance[0, 0] + self.covariance[1, 1]))
        adaptive = self.gate_sigma * position_std
        return float(np.clip(adaptive, self.min_gate_px, self.tight_gate_px))

    def _match_tight(
        self,
        points: np.ndarray,
        scores: np.ndarray,
        distances: np.ndarray,
    ) -> Optional[int]:
        in_gate = np.flatnonzero(distances <= self._tight_limit())
        if len(in_gate) == 0:
            return None
        costs = distances[in_gate] - self.confidence_weight * scores[in_gate]
        return int(in_gate[int(np.argmin(costs))])

    def _kalman_update(self, measurement: np.ndarray) -> None:
        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        innovation = measurement - observation @ self.state
        innovation_cov = observation @ self.covariance @ observation.T
        innovation_cov += np.eye(2) * self.measurement_std ** 2
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(4) - gain @ observation) @ self.covariance
        self.last_measurement = np.asarray(measurement, dtype=np.float64).copy()
        self.updates += 1
        self.missed = 0
        self._pending = []

    def _confirm_pending(
        self,
        points: np.ndarray,
        scores: np.ndarray,
    ) -> Optional[np.ndarray]:
        if not self._pending:
            return None
        best_distance = self.confirm_px
        best_index = None
        best_origin = None
        for origin, predicted in self._pending:
            distances = np.linalg.norm(points - predicted, axis=1)
            index = int(np.argmin(distances))
            # A slightly farther point with much higher confidence can be the ball.
            adjusted = distances[index] - self.confidence_weight * scores[index]
            if distances[index] <= self.confirm_px and adjusted <= best_distance:
                best_distance = adjusted
                best_index = index
                best_origin = origin
        self._pending = []
        if best_index is None or best_origin is None:
            return None
        velocity = (points[best_index] - best_origin) / self.dt
        self._initialize(points[best_index], velocity)
        self.updates = 2
        return self.state[:2].copy()

    def _remember_wide_candidates(
        self,
        points: np.ndarray,
        scores: np.ndarray,
        distances: np.ndarray,
    ) -> None:
        if self.last_measurement is None:
            return
        eligible = (
            (distances <= self.wide_gate_px)
            & (scores >= self.wide_confidence)
        )
        self._pending = []
        for measurement in points[eligible]:
            velocity = measurement - self.last_measurement
            self._pending.append((measurement.copy(), measurement + velocity))

    def _coast(self) -> Optional[np.ndarray]:
        self.missed += 1
        if self.missed > self.max_missed:
            self.state = None
            self.covariance = None
            self.updates = 0
            self.last_measurement = None
            self._pending = []
            return None
        return None if self.state is None else self.state[:2].copy()
