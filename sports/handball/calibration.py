from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


def court_correspondences(
    image_xy: np.ndarray,
    court_xy: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair detected keypoints with court vertices, dropping weak points.

    YOLO marks a missing keypoint as the origin when the model has no
    visibility scores. A reported confidence is a better test: a corner that
    really sits on the first pixel of the frame is kept.
    """
    image_points = np.asarray(image_xy, dtype=np.float32)
    court_points = np.asarray(court_xy, dtype=np.float32)
    if image_points.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
    if image_points.ndim == 3:
        image_points = image_points[0]
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("Image keypoints must have shape (K, 2) or (1, K, 2).")

    count = min(len(image_points), len(court_points))
    image_points = image_points[:count]
    court_points = court_points[:count]
    valid = np.isfinite(image_points).all(axis=1) & np.isfinite(court_points).all(axis=1)

    if confidence is not None:
        scores = np.asarray(confidence, dtype=np.float32).reshape(-1)
        if len(scores) >= count:
            valid &= scores[:count] >= confidence_threshold
        else:
            valid &= _origin_is_missing(image_points)
    else:
        valid &= _origin_is_missing(image_points)

    return image_points[valid], court_points[valid]


def _origin_is_missing(image_points: np.ndarray) -> np.ndarray:
    return (image_points[:, 0] > 1) & (image_points[:, 1] > 1)


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    shaped = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(shaped, matrix).reshape(-1, 2)


def _pixel_errors(
    court_to_image: np.ndarray,
    image_points: np.ndarray,
    court_points: np.ndarray,
) -> np.ndarray:
    projected = _project(court_to_image, court_points)
    return np.linalg.norm(projected - image_points, axis=1)


def _hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
    return float(cv2.contourArea(hull))


def _fit_homography(
    image_points: np.ndarray,
    court_points: np.ndarray,
    method: int,
    ransac_reproj_threshold_px: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(image_points) < 4:
        return None, None
    # Error is measured in the destination frame. Destination is the image so
    # the threshold stays in pixels at every camera zoom.
    matrix, mask = cv2.findHomography(
        court_points,
        image_points,
        method,
        ransac_reproj_threshold_px,
        maxIters=5000,
        confidence=0.999,
    )
    if matrix is None or mask is None:
        return None, None
    return matrix, mask.ravel().astype(bool)


@dataclass
class CourtProjection:
    """Maps image pixels onto the court, in centimetres."""

    image_to_court: np.ndarray
    court_to_image: np.ndarray
    inlier_count: int
    rmse_px: float

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points is None or np.size(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        shaped = np.asarray(points, dtype=np.float32)
        if shaped.ndim != 2 or shaped.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")
        return _project(self.image_to_court, shaped).astype(np.float32)


def estimate_court_projection(
    image_xy: np.ndarray,
    court_xy: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.35,
    ransac_reproj_threshold_px: float = 4.0,
    max_inlier_rmse_px: float = 5.0,
    min_inliers: int = 4,
    min_hull_area_cm2: float = 3_600_000,
) -> Optional[CourtProjection]:
    """Estimate an image-to-court homography that rejects outlier keypoints.

    A single misplaced keypoint can slide every player by metres when every
    point is used in an ordinary least-squares fit. USAC-MAGSAC drops those
    points, then a least-squares fit on the inliers tightens the result.
    The fit is discarded when the inliers collapse onto a small patch or the
    remaining error is still large.
    """
    image_points, court_points = court_correspondences(
        image_xy,
        court_xy,
        confidence,
        confidence_threshold,
    )
    if len(image_points) < min_inliers:
        return None

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    court_to_image, inliers = _fit_homography(
        image_points,
        court_points,
        method,
        ransac_reproj_threshold_px,
    )
    if court_to_image is None and method != cv2.RANSAC:
        court_to_image, inliers = _fit_homography(
            image_points,
            court_points,
            cv2.RANSAC,
            ransac_reproj_threshold_px,
        )
    if court_to_image is None or inliers is None or int(inliers.sum()) < min_inliers:
        return None

    inlier_image = image_points[inliers]
    inlier_court = court_points[inliers]
    refined, _ = _fit_homography(inlier_image, inlier_court, 0, ransac_reproj_threshold_px)
    if refined is not None:
        refined_error = _pixel_errors(refined, inlier_image, inlier_court)
        robust_error = _pixel_errors(court_to_image, inlier_image, inlier_court)
        if float(np.mean(refined_error ** 2)) <= float(np.mean(robust_error ** 2)):
            court_to_image = refined

    if _hull_area(inlier_court) < min_hull_area_cm2:
        return None

    errors = _pixel_errors(court_to_image, inlier_image, inlier_court)
    rmse_px = float(np.sqrt(np.mean(errors ** 2)))
    if rmse_px > max_inlier_rmse_px:
        return None

    try:
        image_to_court = np.linalg.inv(court_to_image)
    except np.linalg.LinAlgError:
        return None

    return CourtProjection(
        image_to_court=image_to_court,
        court_to_image=court_to_image,
        inlier_count=int(inliers.sum()),
        rmse_px=rmse_px,
    )


class CourtCalibrator:
    """Frame-to-frame court calibration for a broadcast camera.

    Keypoint detectors jitter by a pixel or two even when the camera is
    fixed. Smoothing the projected court in the image, then refitting, keeps
    a standing player's court position from wandering. A real camera move is
    left unsmoothed so the radar does not lag behind a pan.
    """

    def __init__(
        self,
        court_xy: np.ndarray,
        alpha: float = 0.35,
        stable_shift_px: float = 25.0,
        max_hold_frames: int = 10,
        **estimate_kwargs,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        self.court_xy = np.asarray(court_xy, dtype=np.float32)
        self.alpha = alpha
        self.stable_shift_px = stable_shift_px
        self.max_hold_frames = max_hold_frames
        self.estimate_kwargs = estimate_kwargs
        self.projection: Optional[CourtProjection] = None
        self._projected_vertices: Optional[np.ndarray] = None
        self._hold_frames = 0

    def update(
        self,
        image_xy: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> Optional[CourtProjection]:
        projection = estimate_court_projection(
            image_xy,
            self.court_xy,
            confidence,
            **self.estimate_kwargs,
        )
        if projection is None:
            self._hold_frames += 1
            if self.projection is not None and self._hold_frames <= self.max_hold_frames:
                return self.projection
            self.projection = None
            self._projected_vertices = None
            return None

        projected = _project(projection.court_to_image, self.court_xy)
        if self._projected_vertices is not None:
            shift = float(np.mean(np.linalg.norm(
                projected - self._projected_vertices, axis=1
            )))
            blend = min(1.0, max(self.alpha, shift / self.stable_shift_px))
            if blend < 1.0:
                projected = (1.0 - blend) * self._projected_vertices + blend * projected
                smoothed, _ = cv2.findHomography(self.court_xy, projected, 0)
                if smoothed is not None:
                    try:
                        image_to_court = np.linalg.inv(smoothed)
                    except np.linalg.LinAlgError:
                        image_to_court = None
                    if image_to_court is not None:
                        projection = CourtProjection(
                            image_to_court=image_to_court,
                            court_to_image=smoothed,
                            inlier_count=projection.inlier_count,
                            rmse_px=projection.rmse_px,
                        )
                        projected = _project(smoothed, self.court_xy)

        self._projected_vertices = projected
        self._hold_frames = 0
        self.projection = projection
        return projection


def inside_court(
    court_xy: np.ndarray,
    length_cm: float,
    width_cm: float,
    margin_cm: float = 200.0,
) -> np.ndarray:
    """True for points on the court or in the surrounding safety zone."""
    points = np.asarray(court_xy, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (points[:, 0] >= -margin_cm)
        & (points[:, 0] <= length_cm + margin_cm)
        & (points[:, 1] >= -margin_cm)
        & (points[:, 1] <= width_cm + margin_cm)
    )
