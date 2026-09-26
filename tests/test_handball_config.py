import unittest

from sports.handball.config import CourtConfiguration, HandballCourtConfiguration


class HandballCourtConfigurationTest(unittest.TestCase):
    def test_default_court_dimensions(self):
        config = HandballCourtConfiguration()

        self.assertEqual(config.length, 4000)
        self.assertEqual(config.width, 2000)
        self.assertEqual(config.goal_width, 300)
        self.assertEqual(config.throw_off_area_radius, 200)
        self.assertEqual(config.free_throw_line_segment_length, 15)
        self.assertEqual(config.free_throw_line_gap_length, 15)

    def test_vertices_and_metadata_have_matching_lengths(self):
        config = CourtConfiguration()

        self.assertEqual(len(config.vertices), 39)
        self.assertEqual(len(config.labels), len(config.vertices))
        self.assertEqual(len(config.colors), len(config.vertices))

    def test_key_indexes_use_existing_vertices(self):
        config = CourtConfiguration()
        vertex_count = len(config.vertices)

        for indexes in [
            config.court_corner_indexes,
            config.left_goal_indexes,
            config.right_goal_indexes,
            config.left_goal_area_indexes,
            config.right_goal_area_indexes,
        ]:
            self.assertTrue(all(1 <= index <= vertex_count for index in indexes))

    def test_free_throw_line_intersects_sideline(self):
        config = CourtConfiguration()
        left_top_free_throw_vertex = config.vertices[18]
        right_top_free_throw_vertex = config.vertices[22]

        self.assertEqual(left_top_free_throw_vertex[1], 0)
        self.assertEqual(right_top_free_throw_vertex[1], 0)
        self.assertAlmostEqual(
            left_top_free_throw_vertex[0],
            config.length - right_top_free_throw_vertex[0],
        )

    def test_ihf_outer_diagonals(self):
        config = CourtConfiguration()
        full_diagonal = _distance((0, 0), (config.length, config.width))
        half_diagonal = _distance((0, 0), (config.center_x, config.width))

        # Guidelines: outer corner to opposite outer corner is 44.72 m,
        # and a half-court diagonal is 28.28 m.
        self.assertAlmostEqual(full_diagonal / 100, 44.72, places=2)
        self.assertAlmostEqual(half_diagonal / 100, 28.28, places=2)

    def test_ihf_goal_area_construction(self):
        config = CourtConfiguration()
        vertices = config.vertices
        left_goal_line_intersections = (vertices[10], vertices[13])
        straight_section = (vertices[11], vertices[12])

        self.assertAlmostEqual(
            _distance(*left_goal_line_intersections),
            1500,
            places=4,
        )
        self.assertAlmostEqual(_distance(*straight_section), config.goal_width, places=4)
        self.assertAlmostEqual(straight_section[0][0], config.goal_area_radius, places=4)

    def test_ihf_free_throw_and_penalty_marks(self):
        config = CourtConfiguration()
        vertices = config.vertices
        post = (0, config.goal_top_y)

        self.assertAlmostEqual(
            _distance(post, vertices[18]),
            config.free_throw_radius,
            places=4,
        )
        self.assertAlmostEqual(
            _distance(post, vertices[19]),
            config.free_throw_radius,
            places=4,
        )
        self.assertAlmostEqual(
            vertices[19][0] - vertices[11][0],
            config.free_throw_radius - config.goal_area_radius,
        )
        self.assertAlmostEqual(vertices[26][0], config.seven_meter_line_distance)
        self.assertAlmostEqual(_distance(vertices[26], vertices[27]), config.seven_meter_line_length)
        self.assertAlmostEqual(vertices[30][0], config.goalkeeper_restraining_line_distance)
        self.assertAlmostEqual(
            _distance(vertices[30], vertices[31]),
            config.goalkeeper_restraining_line_length,
        )
        self.assertEqual(config.throw_off_area_radius * 2, 400)
        self.assertAlmostEqual(abs(vertices[35][0] - config.center_x), config.substitution_line_distance)

    def test_vertices_are_symmetric(self):
        config = CourtConfiguration()
        vertices = config.vertices

        for x, y in vertices:
            mirrored = (config.length - x, y)
            self.assertTrue(
                any(_distance(mirrored, other) < 1e-6 for other in vertices),
                mirrored,
            )
            mirrored_y = (x, config.width - y)
            self.assertTrue(
                any(_distance(mirrored_y, other) < 1e-6 for other in vertices),
                mirrored_y,
            )


def _distance(start, end) -> float:
    return ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5


if __name__ == "__main__":
    unittest.main()
