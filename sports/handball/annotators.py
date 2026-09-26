from math import asin, degrees
from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from sports.handball.config import HandballCourtConfiguration


def _to_pixel(
    point: Tuple[float, float],
    scale: float,
    padding: int,
) -> Tuple[int, int]:
    return (
        int(round(point[0] * scale + padding)),
        int(round(point[1] * scale + padding)),
    )


def _arc_point(
    center: Tuple[float, float],
    radius: float,
    angle_degrees: float,
) -> Tuple[float, float]:
    angle = np.deg2rad(angle_degrees)
    return (
        center[0] + radius * np.cos(angle),
        center[1] + radius * np.sin(angle),
    )


def _draw_arc(
    image: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    start_degrees: float,
    end_degrees: float,
    color: Tuple[int, int, int],
    thickness: int,
    scale: float,
    padding: int,
) -> None:
    center_px = _to_pixel(center, scale, padding)
    radius_px = int(round(radius * scale))
    cv2.ellipse(
        img=image,
        center=center_px,
        axes=(radius_px, radius_px),
        angle=0,
        startAngle=int(round(start_degrees)),
        endAngle=int(round(end_degrees)),
        color=color,
        thickness=thickness,
    )


def _point_at_distance(
    polyline: np.ndarray,
    cumulative: np.ndarray,
    distance: float,
) -> np.ndarray:
    distance = float(np.clip(distance, 0.0, cumulative[-1]))
    index = int(np.searchsorted(cumulative, distance, side="right") - 1)
    index = min(max(index, 0), len(polyline) - 2)
    segment_length = float(cumulative[index + 1] - cumulative[index])
    if segment_length <= 1e-12:
        return polyline[index]
    blend = (distance - cumulative[index]) / segment_length
    return polyline[index] * (1.0 - blend) + polyline[index + 1] * blend


def split_dashes(
    polyline: np.ndarray,
    dash_length: float,
    gap_length: float,
) -> List[np.ndarray]:
    """Split a polyline into dashes measured along its arc length.

    IHF free-throw segments and the spaces between them are each 15 cm.
    Sampling an arc in 1 degree steps overshoots that length, so dashes are
    cut from the cumulative distance instead.
    """
    if dash_length <= 0 or gap_length < 0:
        raise ValueError("Dash length must be positive and gap length non-negative.")

    points = np.asarray(polyline, dtype=float)
    if len(points) < 2:
        return []

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    if total <= 1e-9:
        return []

    dashes = []
    cursor = 0.0
    while cursor < total - 1e-6:
        dash_end = min(cursor + dash_length, total)
        span = [_point_at_distance(points, cumulative, cursor)]
        for index, distance in enumerate(cumulative):
            if cursor + 1e-6 < distance < dash_end - 1e-6:
                span.append(points[index])
        span.append(_point_at_distance(points, cumulative, dash_end))
        dashes.append(np.asarray(span, dtype=float))
        cursor = dash_end + gap_length
    return dashes


def _sample_arc(
    center: Tuple[float, float],
    radius: float,
    start_degrees: float,
    end_degrees: float,
    step_cm: float = 2.0,
) -> np.ndarray:
    sweep = abs(end_degrees - start_degrees)
    arc_length = abs(np.deg2rad(sweep) * radius)
    sample_count = max(2, int(np.ceil(arc_length / step_cm)) + 1)
    angles = np.linspace(start_degrees, end_degrees, sample_count)
    return np.asarray(
        [_arc_point(center, radius, float(angle)) for angle in angles],
        dtype=float,
    )


def _concatenate_polylines(parts: List[np.ndarray]) -> np.ndarray:
    merged = [np.asarray(parts[0], dtype=float)]
    for part in parts[1:]:
        part = np.asarray(part, dtype=float)
        if np.linalg.norm(merged[-1][-1] - part[0]) < 1e-3:
            merged.append(part[1:])
        else:
            merged.append(part)
    return np.vstack(merged)


def free_throw_polyline(
    config: HandballCourtConfiguration,
    side: str,
) -> np.ndarray:
    """Sample one 9 m line from the top sideline to the bottom sideline.

    The arc and the 3 m straight section share one path so the 15 cm dash
    pattern does not restart at the junctions.
    """
    ratio = np.clip(config.goal_top_y / config.free_throw_radius, -1.0, 1.0)
    sideline_angle = degrees(asin(float(ratio)))
    if side == "left":
        center_x = 0.0
        top_angles = (360.0 - sideline_angle, 360.0)
        bottom_angles = (0.0, sideline_angle)
    elif side == "right":
        center_x = float(config.length)
        top_angles = (180.0 + sideline_angle, 180.0)
        bottom_angles = (180.0, 180.0 - sideline_angle)
    else:
        raise ValueError("side must be 'left' or 'right'.")

    top_center = (center_x, config.goal_top_y)
    bottom_center = (center_x, config.goal_bottom_y)
    direction = 1.0 if side == "left" else -1.0
    straight_x = center_x + direction * config.free_throw_radius
    straight = np.array([
        [straight_x, config.goal_top_y],
        [straight_x, config.goal_bottom_y],
    ], dtype=float)
    return _concatenate_polylines([
        _sample_arc(top_center, config.free_throw_radius, *top_angles),
        straight,
        _sample_arc(bottom_center, config.free_throw_radius, *bottom_angles),
    ])


def _draw_dashed_polyline(
    image: np.ndarray,
    polyline: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    scale: float,
    padding: int,
    dash_length: float,
    gap_length: float,
) -> None:
    for dash in split_dashes(polyline, dash_length, gap_length):
        pixels = np.array(
            [_to_pixel((float(point[0]), float(point[1])), scale, padding) for point in dash],
            dtype=np.int32,
        )
        if len(pixels) >= 2:
            cv2.polylines(
                image,
                [pixels],
                isClosed=False,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )


def _draw_goal_frame(
    image: np.ndarray,
    config: HandballCourtConfiguration,
    side: str,
    color: Tuple[int, int, int],
    thickness: int,
    scale: float,
    padding: int,
) -> None:
    if side == "left":
        points = [
            (-config.goal_depth, config.goal_top_y),
            (0, config.goal_top_y),
            (-config.goal_depth, config.goal_top_y),
            (-config.goal_depth, config.goal_bottom_y),
            (-config.goal_depth, config.goal_bottom_y),
            (0, config.goal_bottom_y),
        ]
    else:
        points = [
            (config.length, config.goal_top_y),
            (config.length + config.goal_depth, config.goal_top_y),
            (config.length + config.goal_depth, config.goal_top_y),
            (config.length + config.goal_depth, config.goal_bottom_y),
            (config.length + config.goal_depth, config.goal_bottom_y),
            (config.length, config.goal_bottom_y),
        ]

    for i in range(0, len(points), 2):
        cv2.line(
            image,
            _to_pixel(points[i], scale, padding),
            _to_pixel(points[i + 1], scale, padding),
            color,
            thickness,
        )


def _draw_substitution_marks(
    image: np.ndarray,
    config: HandballCourtConfiguration,
    color: Tuple[int, int, int],
    thickness: int,
    scale: float,
    padding: int,
) -> None:
    mark_half_length = config.substitution_line_length / 2
    for x in (
        config.center_x - config.substitution_line_distance,
        config.center_x + config.substitution_line_distance,
    ):
        top_start = _to_pixel((x, -mark_half_length), scale, padding)
        top_end = _to_pixel((x, mark_half_length), scale, padding)
        bottom_start = _to_pixel((x, config.width - mark_half_length), scale, padding)
        bottom_end = _to_pixel((x, config.width + mark_half_length), scale, padding)
        cv2.line(image, top_start, top_end, color, thickness)
        cv2.line(image, bottom_start, bottom_end, color, thickness)


def draw_court(
    config: HandballCourtConfiguration,
    background_color: sv.Color = sv.Color(38, 132, 170),
    line_color: sv.Color = sv.Color.WHITE,
    goal_color: sv.Color = sv.Color.RED,
    throw_off_area_color: Optional[sv.Color] = None,
    padding: int = 50,
    line_thickness: int = 4,
    scale: float = 0.1,
) -> np.ndarray:
    """
    Draws a handball court with IHF-standard markings.

    Args:
        config (HandballCourtConfiguration): Configuration object containing the
            dimensions and layout of the court.
        background_color (sv.Color, optional): Color of the court background.
            Defaults to sv.Color(38, 132, 170).
        line_color (sv.Color, optional): Color of the court lines.
            Defaults to sv.Color.WHITE.
        goal_color (sv.Color, optional): Color of the goal frame.
            Defaults to sv.Color.RED.
        throw_off_area_color (Optional[sv.Color], optional): Fill color for
            the throw-off area. If None, only the circle line is drawn.
        padding (int, optional): Padding around the court in pixels.
            Defaults to 50.
        line_thickness (int, optional): Thickness of the court lines in pixels.
            Defaults to 4.
        scale (float, optional): Scaling factor for the court dimensions.
            Defaults to 0.1.

    Returns:
        np.ndarray: Image of the handball court.
    """
    scaled_width = int(round(config.width * scale))
    scaled_length = int(round(config.length * scale))
    court = np.ones(
        (scaled_width + 2 * padding, scaled_length + 2 * padding, 3),
        dtype=np.uint8,
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    bgr_line = line_color.as_bgr()
    bgr_goal = goal_color.as_bgr()
    center = (config.center_x, config.center_y)

    if throw_off_area_color is not None:
        cv2.circle(
            court,
            _to_pixel(center, scale, padding),
            radius=int(round(config.throw_off_area_radius * scale)),
            color=throw_off_area_color.as_bgr(),
            thickness=-1,
        )

    for start, end in config.edges:
        point1 = _to_pixel(config.vertices[start - 1], scale, padding)
        point2 = _to_pixel(config.vertices[end - 1], scale, padding)
        cv2.line(court, point1, point2, bgr_line, line_thickness)

    _draw_goal_frame(
        court, config, "left", bgr_goal, line_thickness, scale, padding
    )
    _draw_goal_frame(
        court, config, "right", bgr_goal, line_thickness, scale, padding
    )

    left_top_goal_center = (0, config.goal_top_y)
    left_bottom_goal_center = (0, config.goal_bottom_y)
    right_top_goal_center = (config.length, config.goal_top_y)
    right_bottom_goal_center = (config.length, config.goal_bottom_y)

    _draw_arc(
        court,
        left_top_goal_center,
        config.goal_area_radius,
        270,
        360,
        bgr_line,
        line_thickness,
        scale,
        padding,
    )
    _draw_arc(
        court,
        left_bottom_goal_center,
        config.goal_area_radius,
        0,
        90,
        bgr_line,
        line_thickness,
        scale,
        padding,
    )
    _draw_arc(
        court,
        right_top_goal_center,
        config.goal_area_radius,
        180,
        270,
        bgr_line,
        line_thickness,
        scale,
        padding,
    )
    _draw_arc(
        court,
        right_bottom_goal_center,
        config.goal_area_radius,
        90,
        180,
        bgr_line,
        line_thickness,
        scale,
        padding,
    )

    gap_px = config.free_throw_line_gap_length * scale
    # A stroke thicker than the 15 cm gap fills the break and hides the marking.
    free_throw_thickness = max(1, min(line_thickness, max(1, int(round(gap_px)) - 1)))
    for side in ("left", "right"):
        _draw_dashed_polyline(
            court,
            free_throw_polyline(config, side),
            bgr_line,
            free_throw_thickness,
            scale,
            padding,
            config.free_throw_line_segment_length,
            config.free_throw_line_gap_length,
        )

    _draw_substitution_marks(
        court, config, bgr_line, line_thickness, scale, padding
    )

    cv2.circle(
        court,
        _to_pixel(center, scale, padding),
        radius=int(round(config.throw_off_area_radius * scale)),
        color=bgr_line,
        thickness=line_thickness,
    )

    return court


def draw_points_on_court(
    config: HandballCourtConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    court: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draws points on a handball court.

    Args:
        config (HandballCourtConfiguration): Configuration object containing the
            dimensions and layout of the court.
        xy (np.ndarray): Array of points to draw as (x, y) coordinates.
        face_color (sv.Color, optional): Color of the point faces.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Color of the point edges.
            Defaults to sv.Color.BLACK.
        radius (int, optional): Radius of the points in pixels.
            Defaults to 10.
        thickness (int, optional): Thickness of the point edges in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the court in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for court coordinates.
            Defaults to 0.1.
        court (Optional[np.ndarray], optional): Existing court image to draw on.

    Returns:
        np.ndarray: Image of the handball court with points drawn on it.
    """
    if court is None:
        court = draw_court(config=config, padding=padding, scale=scale)

    if xy is None or np.size(xy) == 0:
        return court

    for point in np.atleast_2d(xy):
        scaled_point = _to_pixel(tuple(point), scale, padding)
        cv2.circle(
            img=court,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1,
        )
        cv2.circle(
            img=court,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness,
        )

    return court


def draw_paths_on_court(
    config: HandballCourtConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    court: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draws paths on a handball court.

    Args:
        config (HandballCourtConfiguration): Configuration object containing the
            dimensions and layout of the court.
        paths (List[np.ndarray]): List of paths with (x, y) coordinates.
        color (sv.Color, optional): Color of the paths.
            Defaults to sv.Color.WHITE.
        thickness (int, optional): Thickness of the paths in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the court in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for court coordinates.
            Defaults to 0.1.
        court (Optional[np.ndarray], optional): Existing court image to draw on.

    Returns:
        np.ndarray: Image of the handball court with paths drawn on it.
    """
    if court is None:
        court = draw_court(config=config, padding=padding, scale=scale)

    if not paths:
        return court

    for path in paths:
        if path is None or np.size(path) == 0:
            continue

        scaled_path = [
            _to_pixel(tuple(point), scale, padding)
            for point in np.atleast_2d(path)
            if point.size > 0 and not np.isnan(point).any()
        ]

        if len(scaled_path) < 2:
            continue

        for i in range(len(scaled_path) - 1):
            cv2.line(
                img=court,
                pt1=scaled_path[i],
                pt2=scaled_path[i + 1],
                color=color.as_bgr(),
                thickness=thickness,
            )

    return court


def _project_points(
    points: np.ndarray,
    court_to_image: np.ndarray,
) -> np.ndarray:
    shaped = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(shaped, court_to_image)
    return projected.reshape(-1, 2)


def _draw_projected_polyline(
    image: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    if len(points) < 2:
        return
    pixels = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        image,
        [pixels],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def draw_projected_court(
    image: np.ndarray,
    config: HandballCourtConfiguration,
    court_to_image: np.ndarray,
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
) -> np.ndarray:
    """Draw court markings warped by a court-to-image homography.

    A calibration is accurate when these lines sit on the markings in the
    frame. Arcs are sampled in court centimetres and then projected, because
    a homography does not map circles to circles.
    """
    bgr = color.as_bgr()
    for start, end in config.edges:
        segment = _project_points(
            [
                config.vertices[start - 1],
                config.vertices[end - 1],
            ],
            court_to_image,
        )
        _draw_projected_polyline(image, segment, bgr, thickness)

    goal_arcs = (
        ((0.0, config.goal_top_y), 270.0, 360.0),
        ((0.0, config.goal_bottom_y), 0.0, 90.0),
        ((float(config.length), config.goal_top_y), 180.0, 270.0),
        ((float(config.length), config.goal_bottom_y), 90.0, 180.0),
    )
    for center, start_angle, end_angle in goal_arcs:
        arc = _sample_arc(
            center,
            config.goal_area_radius,
            start_angle,
            end_angle,
        )
        _draw_projected_polyline(
            image,
            _project_points(arc, court_to_image),
            bgr,
            thickness,
        )

    for side in ("left", "right"):
        _draw_projected_polyline(
            image,
            _project_points(free_throw_polyline(config, side), court_to_image),
            bgr,
            thickness,
        )

    throw_off = _sample_arc(
        (config.center_x, config.center_y),
        config.throw_off_area_radius,
        0.0,
        360.0,
    )
    _draw_projected_polyline(
        image,
        _project_points(throw_off, court_to_image),
        bgr,
        thickness,
    )
    return image
