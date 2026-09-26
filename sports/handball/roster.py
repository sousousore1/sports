"""On-court handball roster.

A player who is already on the court stays in the roster when the detector
misses them. Each team may have at most six field players and one goalkeeper.
After the opening roster is taken, a player is added or removed only through a
substitution segment on the sideline.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from sports.handball.config import HandballCourtConfiguration


FIELD_PLAYER = "player"
GOALKEEPER = "goalkeeper"
REFEREE = "referee"


def normalize_kind(kind: str) -> str:
    name = str(kind).strip().lower()
    if name in {"goalkeeper", "goalie", "gk"}:
        return GOALKEEPER
    if name in {"referee", "ref", "official"}:
        return REFEREE
    return FIELD_PLAYER


def substitution_segments(config: HandballCourtConfiguration) -> np.ndarray:
    """Both sideline substitution segments, in centimetres.

    Each segment runs from 4.5 m before the centre line to 4.5 m after it.
    Shape is (2, 2, 2): segment, endpoint, coordinate.
    """
    centre = float(config.center_x)
    reach = float(config.substitution_line_distance)
    width = float(config.width)
    return np.array(
        [
            [[centre - reach, 0.0], [centre + reach, 0.0]],
            [[centre - reach, width], [centre + reach, width]],
        ],
        dtype=np.float64,
    )


def distance_to_segments(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Shortest distance from each point to the closest segment."""
    samples = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(samples) == 0:
        return np.zeros((0,), dtype=np.float64)
    best = np.full(len(samples), np.inf, dtype=np.float64)
    for segment in np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2):
        start, end = segment
        edge = end - start
        length_sq = float(np.dot(edge, edge))
        if length_sq <= 1e-9:
            distance = np.linalg.norm(samples - start, axis=1)
        else:
            scale = np.clip(((samples - start) @ edge) / length_sq, 0.0, 1.0)
            projection = start + scale[:, None] * edge
            distance = np.linalg.norm(samples - projection, axis=1)
        best = np.minimum(best, distance)
    return best


def on_court(
    points: np.ndarray,
    config: HandballCourtConfiguration,
    margin_cm: float = 40.0,
) -> np.ndarray:
    samples = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(samples) == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (samples[:, 0] >= -margin_cm)
        & (samples[:, 0] <= float(config.length) + margin_cm)
        & (samples[:, 1] >= -margin_cm)
        & (samples[:, 1] <= float(config.width) + margin_cm)
    )


def torso_colors(image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Mean BGR of the central torso of each box."""
    boxes = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    height, width = image.shape[:2]
    colors = np.zeros((len(boxes), 3), dtype=np.float64)
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        left = int(np.clip(x1 + 0.25 * box_w, 0, width - 1))
        right = int(np.clip(x1 + 0.75 * box_w, left + 1, width))
        top = int(np.clip(y1 + 0.12 * box_h, 0, height - 1))
        bottom = int(np.clip(y1 + 0.42 * box_h, top + 1, height))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        colors[index] = crop.reshape(-1, 3).mean(axis=0)
    return colors


def _luma(color: np.ndarray) -> float:
    vector = np.asarray(color, dtype=np.float64).reshape(-1)[:3]
    return float(vector @ np.array([0.114, 0.587, 0.299]))


def cluster_two_teams(colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split jersey colours into two teams. Team 0 is the darker kit."""
    samples = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    luma = samples @ np.array([0.114, 0.587, 0.299])
    centroids = np.stack([samples[int(np.argmin(luma))], samples[int(np.argmax(luma))]])
    labels = np.zeros(len(samples), dtype=int)
    for _ in range(10):
        distance = np.linalg.norm(samples[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distance, axis=1)
        for team in (0, 1):
            chosen = labels == team
            if np.any(chosen):
                centroids[team] = samples[chosen].mean(axis=0)
    if _luma(centroids[0]) > _luma(centroids[1]):
        centroids = centroids[::-1].copy()
        labels = 1 - labels
    return labels.astype(int), centroids


class JerseyColorTeams:
    """Lock two jersey colours once both kits have been seen."""

    def __init__(self, min_samples: int = 8, min_luma_gap: float = 18.0) -> None:
        self.min_samples = min_samples
        self.min_luma_gap = min_luma_gap
        self.centroids: Optional[np.ndarray] = None
        self._samples: List[np.ndarray] = []

    def observe(self, colors: np.ndarray) -> None:
        if self.centroids is not None:
            return
        samples = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(samples).all(axis=1) & (np.linalg.norm(samples, axis=1) > 1.0)
        if np.any(finite):
            self._samples.append(samples[finite])
        stacked = np.vstack(self._samples) if self._samples else np.zeros((0, 3))
        if len(stacked) < self.min_samples:
            return
        labels, centroids = cluster_two_teams(stacked)
        if abs(_luma(centroids[0]) - _luma(centroids[1])) < self.min_luma_gap:
            return
        if int((labels == 0).sum()) < 2 or int((labels == 1).sum()) < 2:
            return
        self.centroids = centroids

    def assign(self, colors: np.ndarray) -> np.ndarray:
        samples = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if len(samples) == 0:
            return np.zeros((0,), dtype=int)
        if self.centroids is None:
            return np.full(len(samples), -1, dtype=int)
        distance = np.linalg.norm(samples[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distance, axis=1).astype(int)


def suppress_overlaps(
    xyxy: np.ndarray,
    confidence: np.ndarray,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """Keep the higher-confidence box when two detections cover one person."""
    boxes = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=bool)
    order = np.argsort(-scores)
    keep = np.zeros(len(boxes), dtype=bool)
    suppressed = np.zeros(len(boxes), dtype=bool)
    for index in order:
        if suppressed[index]:
            continue
        keep[index] = True
        for other in order:
            if other == index or suppressed[other]:
                continue
            if _iou(boxes[index], boxes[other]) >= iou_threshold:
                suppressed[other] = True
    return keep


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_b = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _project_court_to_image(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    shaped = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(shaped, np.asarray(matrix, dtype=np.float64)).reshape(-1, 2)


@dataclass
class RosterTrack:
    track_id: int
    team_id: int
    kind: str
    xyxy: np.ndarray
    foot_court: np.ndarray
    confidence: float
    hits: int
    missed: int
    confirmed: bool
    velocity: np.ndarray
    mean_confidence: float
    exit_hits: int = 0
    last_inside: Optional[np.ndarray] = None

    def report(self) -> "RosterTrack":
        return RosterTrack(
            track_id=self.track_id,
            team_id=self.team_id,
            kind=self.kind,
            xyxy=self.xyxy.copy(),
            foot_court=self.foot_court.copy(),
            confidence=self.confidence,
            hits=self.hits,
            missed=self.missed,
            confirmed=self.confirmed,
            velocity=self.velocity.copy(),
            mean_confidence=self.mean_confidence,
            exit_hits=self.exit_hits,
            last_inside=None if self.last_inside is None else self.last_inside.copy(),
        )


class HandballRoster:
    """Track the players who are allowed to be on the court."""

    def __init__(
        self,
        config: Optional[HandballCourtConfiguration] = None,
        max_field_players: int = 6,
        max_goalkeepers: int = 1,
        confirm_hits: int = 3,
        warmup_frames: int = 30,
        substitution_gate_cm: float = 180.0,
        inside_margin_cm: float = 40.0,
        match_gate_cm: float = 160.0,
        miss_gate_cm: float = 70.0,
        exit_hits: int = 3,
        velocity_decay: float = 0.85,
        referee_coast: int = 8,
    ) -> None:
        self.config = config or HandballCourtConfiguration()
        self.max_field_players = max_field_players
        self.max_goalkeepers = max_goalkeepers
        self.confirm_hits = confirm_hits
        self.warmup_frames = warmup_frames
        self.substitution_gate_cm = substitution_gate_cm
        self.inside_margin_cm = inside_margin_cm
        self.match_gate_cm = match_gate_cm
        self.miss_gate_cm = miss_gate_cm
        self.exit_hits = exit_hits
        self.velocity_decay = velocity_decay
        self.referee_coast = referee_coast
        self.segments = substitution_segments(self.config)
        self.frame_index = 0
        self._tracks: List[RosterTrack] = []
        self._next_id = 1

    @property
    def tracks(self) -> List[RosterTrack]:
        return [track.report() for track in self._tracks]

    def update(
        self,
        xyxy: np.ndarray,
        confidence: np.ndarray,
        kind: Sequence[str],
        team_id: np.ndarray,
        foot_court: np.ndarray,
        court_to_image: Optional[np.ndarray] = None,
    ) -> List[RosterTrack]:
        boxes = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(confidence, dtype=np.float64).reshape(-1)
        teams = np.asarray(team_id, dtype=int).reshape(-1)
        feet = np.asarray(foot_court, dtype=np.float64).reshape(-1, 2)
        kinds = [normalize_kind(name) for name in kind]
        if not (len(boxes) == len(scores) == len(teams) == len(feet) == len(kinds)):
            raise ValueError("Detection fields must have the same length.")

        finite = np.isfinite(boxes).all(axis=1) & np.isfinite(feet).all(axis=1) & np.isfinite(scores)
        boxes, scores, teams, feet = boxes[finite], scores[finite], teams[finite], feet[finite]
        kinds = [name for name, ok in zip(kinds, finite) if ok]

        if len(boxes):
            keep = suppress_overlaps(boxes, scores)
            boxes, scores, teams, feet = boxes[keep], scores[keep], teams[keep], feet[keep]
            kinds = [name for name, ok in zip(kinds, keep) if ok]

        matched_tracks = set()
        matched_detections = set()
        pairs = []
        for track_index, track in enumerate(self._tracks):
            gate = self.match_gate_cm + self.miss_gate_cm * track.missed
            for detection_index, foot in enumerate(feet):
                distance = float(np.linalg.norm(foot - track.foot_court))
                if distance <= gate:
                    pairs.append((distance, track_index, detection_index))
        pairs.sort()
        for _, track_index, detection_index in pairs:
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            self._apply_detection(
                self._tracks[track_index],
                boxes[detection_index],
                float(scores[detection_index]),
                kinds[detection_index],
                int(teams[detection_index]),
                feet[detection_index],
            )
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        survivors: List[RosterTrack] = []
        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks:
                self._coast(track)
            if self._alive(track):
                survivors.append(track)
        self._tracks = survivors

        birth_order = sorted(
            (index for index in range(len(boxes)) if index not in matched_detections),
            key=lambda index: -float(scores[index]),
        )
        for detection_index in birth_order:
            self._birth(
                boxes[detection_index],
                float(scores[detection_index]),
                kinds[detection_index],
                int(teams[detection_index]),
                feet[detection_index],
            )

        if self.frame_index < self.warmup_frames:
            self._enforce_cap(drop_confirmed=True)
        self._place_coasted_boxes(court_to_image)
        self.frame_index += 1
        return self.tracks

    def _team_counts(self, team: int) -> Tuple[int, int]:
        field = 0
        goalkeepers = 0
        for track in self._tracks:
            if track.team_id != team or track.kind == REFEREE:
                continue
            if track.kind == GOALKEEPER:
                goalkeepers += 1
            else:
                field += 1
        return field, goalkeepers

    def _can_add(self, team: int, kind: str) -> bool:
        if kind == REFEREE or team < 0:
            return kind == REFEREE
        field, goalkeepers = self._team_counts(team)
        if kind == GOALKEEPER:
            return goalkeepers < self.max_goalkeepers and field + goalkeepers < self.max_field_players + self.max_goalkeepers
        return field < self.max_field_players and field + goalkeepers < self.max_field_players + self.max_goalkeepers

    def _resolve_team(self, team: int, kind: str) -> int:
        if kind == REFEREE:
            return -1
        if team >= 0:
            return team
        available = [candidate for candidate in (0, 1) if self._can_add(candidate, kind)]
        if not available:
            return -1
        return min(available, key=lambda candidate: sum(self._team_counts(candidate)))

    def _inside(self, foot: np.ndarray) -> bool:
        return bool(on_court(foot, self.config, self.inside_margin_cm)[0])

    def _near_substitution(self, foot: np.ndarray) -> bool:
        return float(distance_to_segments(foot, self.segments)[0]) <= self.substitution_gate_cm

    def _plausible_goalkeeper(self, foot: np.ndarray) -> bool:
        """A goalkeeper stands in front of a goal, not at centre court."""
        reach = float(self.config.goal_area_radius) + 300.0
        x = float(foot[0])
        return x <= reach or x >= float(self.config.length) - reach

    def _left_through_substitution(self, foot: np.ndarray) -> bool:
        """True when a foot has crossed a sideline inside the substitution zone.

        The landing point may already be past the narrow gate. A player who
        steps out through a goal line stays on the roster.
        """
        if self._inside(foot):
            return False
        beyond_sideline = (
            float(foot[1]) < -self.inside_margin_cm
            or float(foot[1]) > float(self.config.width) + self.inside_margin_cm
        )
        if not beyond_sideline:
            return False
        centre = float(self.config.center_x)
        reach = float(self.config.substitution_line_distance) + self.substitution_gate_cm
        return abs(float(foot[0]) - centre) <= reach

    def _apply_detection(
        self,
        track: RosterTrack,
        box: np.ndarray,
        score: float,
        kind: str,
        team: int,
        foot: np.ndarray,
    ) -> None:
        outside = not self._inside(foot)
        leaves_through_substitution = self._left_through_substitution(foot)
        if outside and not leaves_through_substitution and track.kind != REFEREE:
            self._coast(track)
            return

        track.velocity = foot - track.foot_court
        track.foot_court = foot.astype(np.float64).copy()
        track.xyxy = box.astype(np.float64).copy()
        track.confidence = score
        track.hits += 1
        track.missed = 0
        track.mean_confidence += (score - track.mean_confidence) / track.hits
        if (
            kind == GOALKEEPER
            and track.kind == FIELD_PLAYER
            and self._plausible_goalkeeper(foot)
            and self._team_counts(track.team_id)[1] == 0
        ):
            track.kind = GOALKEEPER
        if track.hits >= self.confirm_hits:
            track.confirmed = True
        if leaves_through_substitution:
            track.exit_hits += 1
        else:
            track.exit_hits = 0
            track.last_inside = track.foot_court.copy()

    def _coast(self, track: RosterTrack) -> None:
        track.missed += 1
        if track.kind == REFEREE:
            return
        predicted = track.foot_court + track.velocity
        track.velocity = track.velocity * self.velocity_decay
        if self._inside(predicted):
            track.foot_court = predicted
            track.last_inside = predicted.copy()
            track.exit_hits = 0
            return
        if self._left_through_substitution(predicted):
            track.foot_court = predicted
            track.exit_hits += 1
            return
        if track.last_inside is not None:
            track.foot_court = track.last_inside.copy()
        track.velocity = np.zeros(2, dtype=np.float64)
        track.exit_hits = 0

    def _alive(self, track: RosterTrack) -> bool:
        if track.kind == REFEREE:
            return track.missed <= self.referee_coast
        if track.exit_hits >= self.exit_hits and self._left_through_substitution(track.foot_court):
            return False
        if not track.confirmed and track.missed >= self.confirm_hits:
            return False
        return True

    def _birth(
        self,
        box: np.ndarray,
        score: float,
        kind: str,
        team: int,
        foot: np.ndarray,
    ) -> None:
        if not self._inside(foot) and kind == REFEREE:
            return
        if kind == GOALKEEPER and not self._plausible_goalkeeper(foot):
            kind = FIELD_PLAYER
        resolved_team = self._resolve_team(team, kind)
        if kind != REFEREE:
            opening = self.frame_index < self.warmup_frames
            if opening:
                if not self._inside(foot):
                    return
            elif not self._near_substitution(foot):
                return
            if not self._can_add(resolved_team, kind):
                return
        track = RosterTrack(
            track_id=self._next_id,
            team_id=resolved_team,
            kind=kind,
            xyxy=box.astype(np.float64).copy(),
            foot_court=foot.astype(np.float64).copy(),
            confidence=score,
            hits=1,
            missed=0,
            confirmed=False,
            velocity=np.zeros(2, dtype=np.float64),
            mean_confidence=score,
            last_inside=foot.astype(np.float64).copy() if self._inside(foot) else None,
        )
        if track.hits >= self.confirm_hits:
            track.confirmed = True
        self._next_id += 1
        self._tracks.append(track)

    def _enforce_cap(self, drop_confirmed: bool) -> None:
        for team in sorted({track.team_id for track in self._tracks if track.team_id >= 0}):
            while True:
                field, goalkeepers = self._team_counts(team)
                overflow_field = field > self.max_field_players
                overflow_gk = goalkeepers > self.max_goalkeepers
                overflow_total = field + goalkeepers > self.max_field_players + self.max_goalkeepers
                if not (overflow_field or overflow_gk or overflow_total):
                    break
                candidates = []
                for track in self._tracks:
                    if track.team_id != team or track.kind == REFEREE:
                        continue
                    if overflow_gk and not overflow_field and track.kind != GOALKEEPER:
                        continue
                    if overflow_field and not overflow_gk and track.kind != FIELD_PLAYER:
                        continue
                    if track.confirmed and not drop_confirmed:
                        continue
                    candidates.append(track)
                if not candidates:
                    break
                victim = min(candidates, key=lambda track: (track.hits, track.mean_confidence, -track.missed))
                self._tracks.remove(victim)

    def _place_coasted_boxes(self, court_to_image: Optional[np.ndarray]) -> None:
        if court_to_image is None:
            return
        coasting = [track for track in self._tracks if track.missed > 0 and track.kind != REFEREE]
        if not coasting:
            return
        projected = _project_court_to_image(
            court_to_image,
            np.stack([track.foot_court for track in coasting]),
        )
        for track, foot_image in zip(coasting, projected):
            if not np.isfinite(foot_image).all():
                continue
            x1, y1, x2, y2 = track.xyxy
            box_w = max(2.0, float(x2 - x1))
            box_h = max(2.0, float(y2 - y1))
            cx, fy = float(foot_image[0]), float(foot_image[1])
            track.xyxy = np.array(
                [cx - box_w / 2.0, fy - box_h, cx + box_w / 2.0, fy],
                dtype=np.float64,
            )
