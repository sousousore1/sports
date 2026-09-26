import argparse
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    parent for parent in [CURRENT_DIR, *CURRENT_DIR.parents]
    if (parent / "sports").is_dir()
)
sys.path.insert(0, str(REPO_ROOT))

from sports.annotators.handball import (
    draw_court,
    draw_paths_on_court,
    draw_points_on_court,
    draw_projected_court,
)
from sports.configs.handball import HandballCourtConfiguration
from sports.handball.ball import HandballBallTracker
from sports.handball.calibration import CourtCalibrator
from sports.handball.roster import (
    FIELD_PLAYER,
    GOALKEEPER,
    REFEREE,
    HandballRoster,
    JerseyColorTeams,
    normalize_kind,
    torso_colors,
)


PARENT_DIR = str(CURRENT_DIR)
DEFAULT_TARGET_DIR = os.path.join(PARENT_DIR, "data")

PLAYER_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, "data/handball-player-detection.pt"
)
BALL_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, "data/handball-ball-detection.pt"
)
COURT_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, "data/handball-court-keypoint-detection.pt"
)

CONFIG = HandballCourtConfiguration()
TEAM_COLORS = ["#E53935", "#1E88E5", "#FDD835"]

VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.WHITE,
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(TEAM_COLORS),
    thickness=2,
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(TEAM_COLORS),
    text_color=sv.Color.WHITE,
    text_padding=5,
    text_thickness=1,
)
BALL_ANNOTATOR = sv.CircleAnnotator(
    color=sv.Color.from_hex(TEAM_COLORS[2]),
    thickness=2,
)
PLAYER_CONFIDENCE_THRESHOLD = 0.35


class Mode(Enum):
    """
    Enum class representing different modes for Handball AI examples.
    """

    COURT_RENDERING = "COURT_RENDERING"
    POINT_RENDERING = "POINT_RENDERING"
    PATH_RENDERING = "PATH_RENDERING"
    COURT_DETECTION = "COURT_DETECTION"
    PLAYER_DETECTION = "PLAYER_DETECTION"
    BALL_DETECTION = "BALL_DETECTION"
    RADAR = "RADAR"
    ALL_RENDERINGS = "ALL_RENDERINGS"


def ensure_target_dir(target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)


def require_video_paths(
    source_video_path: Optional[str],
    target_video_path: Optional[str],
    mode: Mode,
) -> Tuple[str, str]:
    if source_video_path is None or target_video_path is None:
        raise ValueError(
            f"{mode.value} requires --source_video_path and --target_video_path."
        )
    return source_video_path, target_video_path


def labels_from_result(result) -> List[str]:
    detections = sv.Detections.from_ultralytics(result)
    names = result.names
    return [names[class_id] for class_id in detections.class_id]


def keypoint_inputs(
    keypoints: sv.KeyPoints,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(keypoints) == 0 or keypoints.xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32), None

    best = 0
    if keypoints.keypoint_confidence is not None and len(keypoints) > 1:
        visible = keypoints.keypoint_confidence >= PLAYER_CONFIDENCE_THRESHOLD
        best = int(np.argmax(visible.sum(axis=1)))

    confidence = None
    if keypoints.keypoint_confidence is not None:
        confidence = keypoints.keypoint_confidence[best]
    return keypoints.xy[best], confidence


def write_video(
    source_video_path: str,
    target_video_path: str,
    frame_generator: Iterator[np.ndarray],
) -> None:
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)


def save_image(target_dir: str, file_name: str, image: np.ndarray) -> str:
    ensure_target_dir(target_dir)
    target_path = os.path.join(target_dir, file_name)
    cv2.imwrite(target_path, image)
    return target_path


def render_court() -> np.ndarray:
    return draw_court(
        config=CONFIG,
        background_color=sv.Color(38, 132, 170),
        line_color=sv.Color.WHITE,
        goal_color=sv.Color.RED,
        padding=80,
        line_thickness=6,
        scale=0.2,
    )


def render_points() -> np.ndarray:
    court = render_court()
    players = {
        0: np.array([
            [650, 750],
            [1050, 1220],
            [1500, 530],
            [2300, 1450],
            [2850, 760],
            [3350, 1200],
        ]),
        1: np.array([
            [700, 1240],
            [1150, 560],
            [1700, 1460],
            [2350, 640],
            [2900, 1300],
            [3300, 840],
        ]),
        2: np.array([[2100, 980]]),
    }

    for team_id, xy in players.items():
        court = draw_points_on_court(
            config=CONFIG,
            xy=xy,
            face_color=sv.Color.from_hex(TEAM_COLORS[team_id]),
            edge_color=sv.Color.WHITE,
            radius=12,
            thickness=3,
            padding=80,
            scale=0.2,
            court=court,
        )

    return court


def render_paths() -> np.ndarray:
    court = render_points()
    paths = [
        np.array([[650, 750], [950, 840], [1280, 760], [1600, 900], [2100, 980]]),
        np.array([[3300, 840], [3020, 900], [2700, 1040], [2400, 980], [2100, 980]]),
        np.array([[1050, 1220], [1350, 1120], [1680, 1240], [1950, 1100]]),
    ]

    return draw_paths_on_court(
        config=CONFIG,
        paths=paths,
        color=sv.Color.WHITE,
        thickness=4,
        padding=80,
        scale=0.2,
        court=court,
    )


def run_court_detection(
    source_video_path: str,
    device: str,
    model_path: str,
) -> Iterator[np.ndarray]:
    court_detection_model = YOLO(model_path).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    calibrator = CourtCalibrator(np.array(CONFIG.vertices, dtype=np.float32))

    for frame in frame_generator:
        result = court_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        label_count = min(len(CONFIG.labels), keypoints.xy.shape[1] if keypoints.xy.ndim == 3 else 0)
        image_xy, confidence = keypoint_inputs(keypoints)
        projection = calibrator.update(image_xy, confidence)

        annotated_frame = frame.copy()
        if len(keypoints) > 0:
            annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
                annotated_frame,
                keypoints,
                CONFIG.labels[:label_count],
            )
        if projection is not None:
            annotated_frame = draw_projected_court(
                annotated_frame,
                CONFIG,
                projection.court_to_image,
                color=sv.Color.from_hex("#00FF87"),
                thickness=2,
            )
        yield annotated_frame


def run_player_detection(
    source_video_path: str,
    device: str,
    model_path: str,
) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(model_path).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        labels = labels_from_result(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels
        )
        yield annotated_frame


def run_ball_detection(
    source_video_path: str,
    device: str,
    model_path: str,
) -> Iterator[np.ndarray]:
    ball_detection_model = YOLO(model_path).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = HandballBallTracker()

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update_detections(detections)

        annotated_frame = frame.copy()
        annotated_frame = BALL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def _track_label(track) -> str:
    if track.kind == GOALKEEPER:
        prefix = "GK"
    elif track.kind == REFEREE:
        prefix = "Ref"
    else:
        prefix = "P"
    return f"{prefix}{track.track_id}"


def run_radar(
    source_video_path: str,
    device: str,
    player_model_path: str,
    court_model_path: str,
) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(player_model_path).to(device=device)
    court_detection_model = YOLO(court_model_path).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    calibrator = CourtCalibrator(np.array(CONFIG.vertices, dtype=np.float32))
    roster = HandballRoster(CONFIG)
    jerseys = JerseyColorTeams()

    for frame in frame_generator:
        court_result = court_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(court_result)
        image_xy, confidence = keypoint_inputs(keypoints)
        projection = calibrator.update(image_xy, confidence)

        player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(player_result)
        if detections.confidence is not None:
            detections = detections[detections.confidence >= PLAYER_CONFIDENCE_THRESHOLD]

        annotated_frame = frame.copy()
        if projection is None:
            if len(detections):
                labels = [
                    player_result.names[class_id] for class_id in detections.class_id
                ]
                annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
                annotated_frame = BOX_LABEL_ANNOTATOR.annotate(
                    annotated_frame, detections, labels=labels
                )
            yield annotated_frame
            continue

        if len(detections) == 0:
            kinds = []
            teams = np.zeros((0,), dtype=int)
            scores = np.zeros((0,), dtype=np.float64)
            feet_court = np.zeros((0, 2), dtype=np.float64)
            boxes = np.zeros((0, 4), dtype=np.float64)
        else:
            kinds = [
                normalize_kind(player_result.names[int(class_id)])
                for class_id in detections.class_id
            ]
            feet_image = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            feet_court = projection.transform_points(points=feet_image)
            boxes = detections.xyxy
            if detections.confidence is None:
                scores = np.ones(len(detections), dtype=np.float64)
            else:
                scores = np.asarray(detections.confidence, dtype=np.float64)
            colors = torso_colors(frame, boxes)
            field_colors = [
                colors[index]
                for index, kind in enumerate(kinds)
                if kind == FIELD_PLAYER
            ]
            if field_colors:
                jerseys.observe(np.stack(field_colors))
            teams = jerseys.assign(colors)
            for index, kind in enumerate(kinds):
                if kind == REFEREE:
                    teams[index] = -1

        tracks = roster.update(
            xyxy=boxes,
            confidence=scores,
            kind=kinds,
            team_id=teams,
            foot_court=feet_court,
            court_to_image=projection.court_to_image,
        )
        visible = [track for track in tracks if track.confirmed or track.hits >= 2]
        if visible:
            boxes = np.stack([track.xyxy for track in visible]).astype(np.float32)
            roster_detections = sv.Detections(
                xyxy=boxes,
                confidence=np.array([track.confidence for track in visible], dtype=np.float32),
                class_id=np.array([
                    track.team_id if track.team_id >= 0 else 2 for track in visible
                ]),
            )
            annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, roster_detections)
            annotated_frame = BOX_LABEL_ANNOTATOR.annotate(
                annotated_frame,
                roster_detections,
                labels=[_track_label(track) for track in visible],
            )

        radar = draw_court(config=CONFIG)
        for team_id, color in ((0, TEAM_COLORS[0]), (1, TEAM_COLORS[1])):
            points = [
                track.foot_court
                for track in visible
                if track.team_id == team_id and track.kind != REFEREE
            ]
            if not points:
                continue
            radar = draw_points_on_court(
                config=CONFIG,
                xy=np.array(points, dtype=np.float32),
                face_color=sv.Color.from_hex(color),
                edge_color=sv.Color.WHITE,
                radius=16,
                court=radar,
            )

        h, w, _ = frame.shape
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h,
        )
        yield sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)


def run_rendering_mode(target_dir: str, mode: Mode) -> Dict[str, str]:
    renderers = {
        Mode.COURT_RENDERING: ("handball-court.png", render_court),
        Mode.POINT_RENDERING: ("handball-points.png", render_points),
        Mode.PATH_RENDERING: ("handball-paths.png", render_paths),
    }

    if mode == Mode.ALL_RENDERINGS:
        selected_modes = [
            Mode.COURT_RENDERING,
            Mode.POINT_RENDERING,
            Mode.PATH_RENDERING,
        ]
    else:
        selected_modes = [mode]

    output_paths = {}
    for selected_mode in selected_modes:
        file_name, renderer = renderers[selected_mode]
        output_paths[selected_mode.value] = save_image(
            target_dir=target_dir,
            file_name=file_name,
            image=renderer(),
        )

    return output_paths


def main(
    source_video_path: Optional[str],
    target_video_path: Optional[str],
    target_dir: str,
    device: str,
    mode: Mode,
    player_model_path: str,
    ball_model_path: str,
    court_model_path: str,
) -> Optional[Dict[str, str]]:
    if mode in {
        Mode.COURT_RENDERING,
        Mode.POINT_RENDERING,
        Mode.PATH_RENDERING,
        Mode.ALL_RENDERINGS,
    }:
        return run_rendering_mode(target_dir=target_dir, mode=mode)

    source_video_path, target_video_path = require_video_paths(
        source_video_path, target_video_path, mode
    )

    if mode == Mode.COURT_DETECTION:
        frame_generator = run_court_detection(
            source_video_path=source_video_path,
            device=device,
            model_path=court_model_path,
        )
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path,
            device=device,
            model_path=player_model_path,
        )
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path,
            device=device,
            model_path=ball_model_path,
        )
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path,
            device=device,
            player_model_path=player_model_path,
            court_model_path=court_model_path,
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    write_video(
        source_video_path=source_video_path,
        target_video_path=target_video_path,
        frame_generator=frame_generator,
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--source_video_path", type=str)
    parser.add_argument("--target_video_path", type=str)
    parser.add_argument("--target_dir", type=str, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=Mode, default=Mode.COURT_RENDERING)
    parser.add_argument(
        "--player_model_path", type=str, default=PLAYER_DETECTION_MODEL_PATH
    )
    parser.add_argument("--ball_model_path", type=str, default=BALL_DETECTION_MODEL_PATH)
    parser.add_argument(
        "--court_model_path", type=str, default=COURT_DETECTION_MODEL_PATH
    )
    args = parser.parse_args()

    paths = main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        target_dir=args.target_dir,
        device=args.device,
        mode=args.mode,
        player_model_path=args.player_model_path,
        ball_model_path=args.ball_model_path,
        court_model_path=args.court_model_path,
    )
    if paths is not None:
        for mode, target_path in paths.items():
            print(f"{mode}: {target_path}")
