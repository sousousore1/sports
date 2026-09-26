"""Ground-plane camera for a handball broadcast.

The motion board reads the camera in metres from one goal line and one
sideline, plus a height. A negative length is a camera behind that goal.
The frame does not see the whole 40 m by 20 m floor: its border meets the
floor on a line, and that line is the edge of the shooting range.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from sports.handball.config import HandballCourtConfiguration


@dataclass(frozen=True)
class CourtCamera:
    """Camera pose above the court plane. Distances are centimetres."""

    x_cm: float
    y_cm: float
    height_cm: float
    focal_px: float

    @property
    def x_m(self) -> float:
        return self.x_cm / 100.0

    @property
    def y_m(self) -> float:
        return self.y_cm / 100.0

    @property
    def height_m(self) -> float:
        return self.height_cm / 100.0

    def readout(self, player_count: int) -> str:
        return (
            f"{player_count} 人を表示・カメラ "
            f"({self.x_m:.1f}, {self.y_m:.1f}) 高さ {self.height_m:.1f} m"
        )


def look_at_rotation(camera_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    """World-to-camera rotation. Rows are the camera axes in world coordinates.

    The camera looks along its +Z axis. World +Z is up, so image +Y points down.
    """
    camera = np.asarray(camera_xyz, dtype=np.float64).reshape(3)
    target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    forward = target - camera
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        raise ValueError("The camera target must not be the camera position.")
    forward = forward / forward_norm
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        raise ValueError("The camera cannot look straight up or straight down.")
    right = right / right_norm
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


def court_homography(
    camera_xyz: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    focal_px: float,
    image_size: tuple,
) -> np.ndarray:
    """Court (x, y) centimetres to image pixels. The floor is z = 0."""
    width, height = image_size
    focal = float(focal_px)
    intrinsics = np.array(
        [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotation = np.asarray(rotation_world_to_camera, dtype=np.float64)
    camera = np.asarray(camera_xyz, dtype=np.float64).reshape(3, 1)
    projection = intrinsics @ np.hstack([rotation, -rotation @ camera])
    return np.column_stack([projection[:, 0], projection[:, 1], projection[:, 3]])


def _principal_point(image_size: tuple) -> tuple:
    width, height = image_size
    return (width - 1) / 2.0, (height - 1) / 2.0


def _focal_length_px(homography: np.ndarray, image_size: tuple) -> float:
    """Square-pixel focal length that makes the court axes perpendicular."""
    cx, cy = _principal_point(image_size)
    metric = np.array(
        [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [-cx, -cy, cx * cx + cy * cy]],
        dtype=np.float64,
    )
    h1, h2 = homography[:, 0], homography[:, 1]
    orthogonal = float(h1 @ metric @ h2)
    lengths = float(h1 @ metric @ h1 - h2 @ metric @ h2)
    # h1^T ω h2 = 0 and h1^T ω h1 = h2^T ω h2, with ω = metric / f^2 + diag(0, 0, 1).
    candidates = []
    if abs(orthogonal) > 1e-12:
        inverse_focal_sq = -float(h1[2] * h2[2]) / orthogonal
        if inverse_focal_sq > 1e-12:
            candidates.append(1.0 / inverse_focal_sq)
    if abs(lengths) > 1e-12:
        inverse_focal_sq = -float(h1[2] * h1[2] - h2[2] * h2[2]) / lengths
        if inverse_focal_sq > 1e-12:
            candidates.append(1.0 / inverse_focal_sq)
    if not candidates:
        raise ValueError("The homography does not determine a focal length.")
    return float(np.sqrt(np.median(candidates)))


def camera_from_homography(
    court_to_image: np.ndarray,
    image_size: tuple,
) -> CourtCamera:
    """Recover the camera position and height from a court-to-image homography.

    The principal point is the image centre and the pixels are square. The
    court is the plane z = 0, with z pointing up.
    """
    homography = np.asarray(court_to_image, dtype=np.float64).reshape(3, 3)
    if abs(homography[2, 2]) > 1e-9:
        homography = homography / homography[2, 2]
    focal = _focal_length_px(homography, image_size)
    cx, cy = _principal_point(image_size)
    intrinsics = np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    mapped = np.linalg.inv(intrinsics) @ homography
    rotation_xy = mapped[:, :2]
    translation = mapped[:, 2]
    scale = (np.linalg.norm(rotation_xy[:, 0]) + np.linalg.norm(rotation_xy[:, 1])) / 2.0
    if scale < 1e-9:
        raise ValueError("The homography has no court scale.")
    rotation_xy = rotation_xy / scale
    translation = translation / scale
    normal = np.cross(rotation_xy[:, 0], rotation_xy[:, 1])
    rotation = np.column_stack([rotation_xy, normal])
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        rotation = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    centre = -rotation.T @ translation
    if centre[2] < 0:
        centre = -centre
    return CourtCamera(
        x_cm=float(centre[0]),
        y_cm=float(centre[1]),
        height_cm=float(abs(centre[2])),
        focal_px=focal,
    )


def _ground_points(
    image_to_court: np.ndarray,
    pixels: np.ndarray,
    front_pixel: np.ndarray,
) -> np.ndarray:
    """Project pixels onto z = 0. Pixels behind the camera become NaN.

    The horizon is the image/floor intersection. Samples on the far side of
    it are not part of the shooting range. ``front_pixel`` is a pixel on the
    near floor, used only to fix the homography's sign.
    """
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    matrix = np.asarray(image_to_court, dtype=np.float64)
    front = np.array([float(front_pixel[0]), float(front_pixel[1]), 1.0])
    if float(matrix[2] @ front) < 0:
        matrix = -matrix
    homo = np.hstack([pixels, np.ones((len(pixels), 1))])
    mapped = (matrix @ homo.T).T
    depth = mapped[:, 2]
    ground = np.full((len(pixels), 2), np.nan, dtype=np.float64)
    in_front = depth > 1e-9
    ground[in_front] = mapped[in_front, :2] / depth[in_front, None]
    return ground


def visible_court_polygon(
    court_to_image: np.ndarray,
    image_size: tuple,
    config: HandballCourtConfiguration,
    samples_per_edge: int = 32,
) -> np.ndarray:
    """Court-centimetre polygon of the floor that this frame can see.

    Image corners are projected onto z = 0 and clipped to a neighbourhood of
    the court. The edge that crosses the court is the boundary drawn on the
    motion board.
    """
    width, height = image_size
    try:
        image_to_court = np.linalg.inv(np.asarray(court_to_image, dtype=np.float64))
    except np.linalg.LinAlgError:
        return np.zeros((0, 2), dtype=np.float64)

    edges = []
    xs = np.linspace(0, width - 1, samples_per_edge)
    ys = np.linspace(0, height - 1, samples_per_edge)
    edges.append(np.stack([xs, np.zeros_like(xs)], axis=1))
    edges.append(np.stack([np.full_like(ys, width - 1), ys], axis=1))
    edges.append(np.stack([xs[::-1], np.full_like(xs, height - 1)], axis=1))
    edges.append(np.stack([np.zeros_like(ys), ys[::-1]], axis=1))
    border = np.vstack(edges)
    ground = _ground_points(image_to_court, border, ((width - 1) / 2.0, height - 1))
    margin = max(float(config.length), float(config.width))
    finite = [
        point for point in ground
        if np.isfinite(point).all() and abs(point[0]) <= margin * 3 and abs(point[1]) <= margin * 3
    ]
    if len(finite) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    cloud = np.asarray(finite, dtype=np.float32)
    hull = cv2.convexHull(cloud).reshape(-1, 2)
    court = np.array(
        [[0, 0], [config.length, 0], [config.length, config.width], [0, config.width]],
        dtype=np.float32,
    )
    visible = _clip_polygon(hull, court)
    return visible.astype(np.float64)


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman clip of ``subject`` against the convex ``clip``."""
    output = np.asarray(subject, dtype=np.float64)
    clip_points = np.asarray(clip, dtype=np.float64)
    for index, start in enumerate(clip_points):
        end = clip_points[(index + 1) % len(clip_points)]
        if len(output) == 0:
            return output
        input_points = output
        output_list = []
        previous = input_points[-1]
        for current in input_points:
            if _inside(current, start, end):
                if not _inside(previous, start, end):
                    output_list.append(_intersection(previous, current, start, end))
                output_list.append(current)
            elif _inside(previous, start, end):
                output_list.append(_intersection(previous, current, start, end))
            previous = current
        output = np.asarray(output_list, dtype=np.float64) if output_list else np.zeros((0, 2))
    return output


def _inside(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    edge = end - start
    return float(edge[0] * (point[1] - start[1]) - edge[1] * (point[0] - start[0])) >= -1e-6


def _intersection(p0: np.ndarray, p1: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = p1 - p0
    edge = end - start
    denominator = edge[0] * segment[1] - edge[1] * segment[0]
    if abs(denominator) < 1e-9:
        return p1
    scale = ((start[1] - p0[1]) * edge[0] - (start[0] - p0[0]) * edge[1]) / denominator
    return p0 + scale * segment


def analyze_intersections(
    image_xy: np.ndarray,
    court_xy: np.ndarray,
    image_size: tuple,
    config: HandballCourtConfiguration,
):
    """Recover the camera from court-line intersections.

    The motion board starts once four image points are paired with four
    court intersections. The homography gives the camera pose and the part
    of the court this frame can see.
    """
    from sports.handball.calibration import estimate_court_projection

    projection = estimate_court_projection(image_xy, court_xy)
    if projection is None:
        return None
    camera = camera_from_homography(projection.court_to_image, image_size)
    visible = visible_court_polygon(projection.court_to_image, image_size, config)
    return projection, camera, visible
