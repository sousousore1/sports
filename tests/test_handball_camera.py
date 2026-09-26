import unittest

import cv2
import numpy as np

from sports.handball.annotators import draw_court, draw_visible_court
from sports.handball.camera import (
    analyze_intersections,
    camera_from_homography,
    court_homography,
    look_at_rotation,
    visible_court_polygon,
)
from sports.handball.config import HandballCourtConfiguration


CONFIG = HandballCourtConfiguration()
IMAGE_SIZE = (1280, 720)
# Behind the left goal line and outside the near sideline, as on the motion board.
CAMERA_CM = np.array([-780.0, 2390.0, 470.0])
FOCAL_PX = 2200.0


def _sideline_camera():
    target = np.array([2000.0, 1000.0, 0.0])
    rotation = look_at_rotation(CAMERA_CM, target)
    homography = court_homography(CAMERA_CM, rotation, focal_px=FOCAL_PX, image_size=IMAGE_SIZE)
    return homography


def _polygon_area(polygon: np.ndarray) -> float:
    return abs(float(np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - np.roll(polygon[:, 0], -1) * polygon[:, 1]
    ))) / 2.0


class CourtCameraTest(unittest.TestCase):
    def test_pose_round_trip_matches_the_sideline_camera(self):
        homography = _sideline_camera()
        recovered = camera_from_homography(homography, IMAGE_SIZE)
        self.assertLess(abs(recovered.x_cm - CAMERA_CM[0]), 30.0)
        self.assertLess(abs(recovered.y_cm - CAMERA_CM[1]), 30.0)
        self.assertLess(abs(recovered.height_cm - CAMERA_CM[2]), 20.0)
        self.assertLess(abs(recovered.focal_px - FOCAL_PX), 40.0)
        readout = recovered.readout(14)
        self.assertIn("(-7.8, 23.9)", readout)
        self.assertIn("高さ 4.7 m", readout)
        self.assertIn("14 人を表示", readout)

    def test_the_frame_cuts_a_wedge_instead_of_the_whole_court(self):
        visible = visible_court_polygon(_sideline_camera(), IMAGE_SIZE, CONFIG)
        self.assertGreaterEqual(len(visible), 3)
        area = _polygon_area(visible)
        court_area = float(CONFIG.length * CONFIG.width)
        self.assertLess(area, court_area * 0.95)
        self.assertGreater(area, court_area * 0.15)
        # The camera sits outside the near-left corner, so that corner is out of frame.
        self.assertGreater(np.min(np.linalg.norm(visible - np.array([0.0, CONFIG.width]), axis=1)), 80.0)

    def test_pixels_past_the_horizon_do_not_fill_the_court(self):
        camera = np.array([-470.0, 2350.0, 300.0])
        target = np.array([90.0, 1130.0, 0.0])
        homography = court_homography(
            camera, look_at_rotation(camera, target), focal_px=1280.0, image_size=IMAGE_SIZE
        )
        visible = visible_court_polygon(homography, IMAGE_SIZE, CONFIG)
        self.assertGreaterEqual(len(visible), 3)
        area = _polygon_area(visible)
        self.assertLess(area, float(CONFIG.length * CONFIG.width) * 0.85)
        # The far end of the court is outside this frame.
        self.assertGreater(
            np.min(np.linalg.norm(visible - np.array([CONFIG.length, 0.0]), axis=1)),
            200.0,
        )

    def test_the_bottom_of_the_frame_is_closer_than_the_top(self):
        image_to_court = np.linalg.inv(_sideline_camera())

        def ground(u, v):
            point = np.array([[[u, v]]], dtype=np.float32)
            return cv2.perspectiveTransform(point, image_to_court).reshape(2)

        camera_xy = CAMERA_CM[:2]
        bottom = ground(640, 700)
        top = ground(640, 20)
        self.assertLess(
            np.linalg.norm(bottom - camera_xy),
            np.linalg.norm(top - camera_xy),
        )

    def test_four_intersections_recover_the_camera_and_the_wedge(self):
        homography = _sideline_camera()
        grid_x = np.linspace(200, 3800, 8)
        grid_y = np.linspace(200, 1800, 5)
        court = np.array([[x, y] for y in grid_y for x in grid_x], dtype=np.float32)
        image = cv2.perspectiveTransform(court.reshape(-1, 1, 2), homography).reshape(-1, 2)
        inside = (
            (image[:, 0] > 8) & (image[:, 0] < IMAGE_SIZE[0] - 8)
            & (image[:, 1] > 8) & (image[:, 1] < IMAGE_SIZE[1] - 8)
        )
        self.assertGreaterEqual(int(inside.sum()), 4)
        analyzed = analyze_intersections(image[inside], court[inside], IMAGE_SIZE, CONFIG)
        self.assertIsNotNone(analyzed)
        _projection, camera, visible = analyzed
        self.assertLess(abs(camera.x_cm - CAMERA_CM[0]), 40.0)
        self.assertLess(abs(camera.y_cm - CAMERA_CM[1]), 40.0)
        self.assertLess(abs(camera.height_cm - CAMERA_CM[2]), 25.0)
        self.assertGreaterEqual(len(visible), 3)
        self.assertLess(_polygon_area(visible), float(CONFIG.length * CONFIG.width) * 0.95)

    def test_the_shooting_range_is_drawn_on_the_full_court(self):
        visible = visible_court_polygon(_sideline_camera(), IMAGE_SIZE, CONFIG)
        court = draw_court(config=CONFIG, padding=120, scale=0.1)
        painted = draw_visible_court(
            config=CONFIG,
            polygon=visible,
            camera_xy=(CAMERA_CM[0], CAMERA_CM[1]),
            padding=120,
            scale=0.1,
            court=court.copy(),
        )
        orange = (
            (painted[:, :, 2] > 160)
            & (painted[:, :, 1] > 80)
            & (painted[:, :, 1] < 210)
            & (painted[:, :, 0] < 120)
        )
        self.assertGreater(int(orange.sum()), 20)
        self.assertFalse(np.array_equal(painted, court))


if __name__ == "__main__":
    unittest.main()
