import numpy as np
import jax.test_util
import ard.utils.geometry as geo_utils
import pytest


class TestGetClosestPoint:
    def setup_method(self):
        self.get_closest_point_jac = jax.jacobian(geo_utils.get_closest_point, [0])
        pass

    def test_get_closest_point_45_deg_with_end(self):

        point = np.array([10, 10])
        line_a = np.array([0, 10])
        line_b = np.array([10, 0])
        line_vector = line_b - line_a

        test_result = geo_utils.get_closest_point(point, line_a, line_b, line_vector)

        assert np.all(test_result == np.array([5, 5]))

    def test_get_closest_point_90_deg_with_end(self):

        point = np.array([0, 0])
        line_a = np.array([0, 10])
        line_b = np.array([10, 10])
        line_vector = line_b - line_a

        test_result = geo_utils.get_closest_point(point, line_a, line_b, line_vector)

        assert np.all(test_result == line_a)

    def test_get_closest_point_gt90_deg_with_end(self):

        point = np.array([0, 0])
        line_a = np.array([5, 5])
        line_b = np.array([10, 5])
        line_vector = line_b - line_a

        test_result = geo_utils.get_closest_point(point, line_a, line_b, line_vector)

        assert np.all(test_result == line_a)

    def test_get_closest_point_180_deg_with_end(self):

        point = np.array([0, 5])
        line_a = np.array([5, 5])
        line_b = np.array([10, 5])
        line_vector = line_b - line_a

        test_result = geo_utils.get_closest_point(point, line_a, line_b, line_vector)

        assert np.all(test_result == line_a)

    def test_get_closest_point_point_on_segment(self):
        """
        Test for a point exactly on the line segment
        """

        test_point = np.array([3, 3, 3])
        test_start = np.array([0, 0, 0])
        test_end = np.array([5, 5, 5])
        line_vector = test_end - test_start

        test_result = geo_utils.get_closest_point(
            test_point, test_start, test_end, line_vector
        )

        assert np.all(test_result == test_point)

    def test_get_closest_point_point_near_end(self):
        """
        Test for a point near the end of the line segment
        """

        test_point = np.array([6, 6, 6])
        test_start = np.array([0, 0, 0])
        test_end = np.array([5, 5, 5])
        line_vector = test_end - test_start

        test_result = geo_utils.get_closest_point(
            test_point, test_start, test_end, line_vector
        )

        assert np.all(test_result == test_end)

    def test_get_closest_point_point_near_start(self):
        """
        Test for a point near the start of the line segment
        """

        test_point = np.array([-1, -1, -2])
        test_start = np.array([0, 0, 0])
        test_end = np.array([5, 5, 5])
        line_vector = test_end - test_start

        test_result = geo_utils.get_closest_point(
            test_point, test_start, test_end, line_vector
        )

        assert np.all(test_result == test_start)

    def test_get_closest_point_point_near_middle(self):
        """
        Test for a point near the middle of the line segment
        """

        test_point = np.array([5, 5, 2])
        test_start = np.array([0, 0, 0])
        test_end = np.array([0, 0, 5])
        line_vector = test_end - test_start

        test_result = geo_utils.get_closest_point(
            test_point, test_start, test_end, line_vector
        )

        assert np.all(test_result == np.array([0, 0, 2]))

    def test_get_closest_point_jac(self):
        """
        Test for gradient for a point near the middle of the line segment
        """

        test_point = np.array([5, 0, 2], dtype=float)
        test_start = np.array([0, 0, 0], dtype=float)
        test_end = np.array([0, 0, 5], dtype=float)
        line_vector = test_end - test_start

        tr_dp = self.get_closest_point_jac(
            test_point, test_start, test_end, line_vector
        )

        assert np.all(tr_dp == np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]))

        try:
            jax.test_util.check_grads(
                geo_utils.get_closest_point,
                (test_point, test_start, test_end, line_vector),
                order=1,
            )
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )


class TestPointToLineSeg:
    def setup_method(self):
        self.distance_point_to_lineseg_nd_grad = jax.grad(
            geo_utils.distance_point_to_lineseg_nd, [0]
        )
        pass

    def test_distance_point_to_lineseg_nd_45_deg_with_end(self):

        point = np.array([10, 10])
        line_a = np.array([0, 10])
        line_b = np.array([10, 0])

        test_result = geo_utils.distance_point_to_lineseg_nd(point, line_a, line_b)

        assert test_result == 7.0710678118654755

    def test_distance_point_to_lineseg_90_deg_with_end(self):

        point = np.array([0, 0])
        line_a = np.array([0, 10])
        line_b = np.array([10, 10])

        test_result = geo_utils.distance_point_to_lineseg_nd(point, line_a, line_b)

        assert test_result == 10.0

    def test_distance_point_to_lineseg_gt90_deg_with_end(self):

        point = np.array([0, 0])
        line_a = np.array([5, 5])
        line_b = np.array([10, 5])

        test_result = geo_utils.distance_point_to_lineseg_nd(point, line_a, line_b)

        assert test_result == 7.0710678118654755

    def test_distance_point_to_lineseg_180_deg_with_end(self):

        point = np.array([0, 5])
        line_a = np.array([5, 5])
        line_b = np.array([10, 5])

        test_result = geo_utils.distance_point_to_lineseg_nd(point, line_a, line_b)

        assert test_result == 5.0

    def test_distance_point_to_lineseg_nd_point_to_point(self):
        """
        Test for a point to point calculation
        """

        test_point = np.array([5, 5, 5])
        test_start = np.array([0, 0, 0])
        test_end = np.array([0, 0, 0])

        test_result = geo_utils.distance_point_to_lineseg_nd(
            test_point, test_start, test_end
        )

        assert test_result == pytest.approx(8.660254037844387)

    def test_distance_point_to_lineseg_nd_point_on_segment(self):
        """
        Test for a point exactly on the line segment
        """

        test_point = np.array([3, 3, 3])
        test_start = np.array([0, 0, 0])
        test_end = np.array([5, 5, 5])

        test_result = geo_utils.distance_point_to_lineseg_nd(
            test_point, test_start, test_end
        )

        assert test_result == pytest.approx(0.0)

    def test_distance_point_to_lineseg_nd_point_near_end(self):
        """
        Test for a point near the end of the line segment
        """

        test_point = np.array([6, 6, 6])
        test_start = np.array([0, 0, 0])
        test_end = np.array([5, 5, 5])

        test_result = geo_utils.distance_point_to_lineseg_nd(
            test_point, test_start, test_end
        )

        assert test_result == pytest.approx(1.7320508075688772)

    def test_distance_point_to_lineseg_nd_point_near_start(self):
        """
        Test for a point near the start of the line segment
        """

        test_point = np.array([-1, -1, -2])
        test_start = np.array([0, 0, 0])
        test_end = np.array([5, 5, 5])

        test_result = geo_utils.distance_point_to_lineseg_nd(
            test_point, test_start, test_end
        )

        assert test_result == pytest.approx(2.449489742783178)

    def test_distance_point_to_lineseg_nd_point_near_middle(self):
        """
        Test for a point near the middle of the line segment
        """

        test_point = np.array([5, 5, 2])
        test_start = np.array([0, 0, 0])
        test_end = np.array([0, 0, 5])

        test_result = geo_utils.distance_point_to_lineseg_nd(
            test_point, test_start, test_end
        )

        assert test_result == pytest.approx(7.0710678118654755)

    def test_distance_point_to_lineseg_nd_grad(self):
        """
        Test for gradient for a point near the middle of the line segment
        """

        test_point = np.array([5, 0, 2], dtype=float)
        test_start = np.array([0, 0, 0], dtype=float)
        test_end = np.array([0, 0, 5], dtype=float)

        tr_dp = self.distance_point_to_lineseg_nd_grad(test_point, test_start, test_end)

        assert np.all(tr_dp == np.array([1, 0, 0]))

        try:
            jax.test_util.check_grads(
                geo_utils.distance_point_to_lineseg_nd,
                (test_point, test_start, test_end),
                order=1,
            )
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )


class TestLineSegToLineSeg:
    def setup_method(self):
        self.distance_lineseg_to_lineseg_nd_grad = jax.grad(
            geo_utils.distance_lineseg_to_lineseg_nd, [0]
        )
        pass

    def test_distance_lineseg_to_lineseg_nd_parallel_2d(self):
        """
        Test distance between line segments 2d for parallel lines
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 5])])
        line_b = np.array([np.array([5, 0]), np.array([5, 5])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(
            5.0, rel=1e-1
        )  # inexact with parallel lines due to smooth max/min

    def test_distance_lineseg_to_lineseg_nd_shared_point_2d(self):
        """
        Test distance between line segments 2d for intersecting lines with a shared point
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 5])])
        line_b = np.array([np.array([0, 0]), np.array([5, 5])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(0.0, abs=1e-2)

    def test_distance_lineseg_to_lineseg_nd_intersect_2d(self):
        """
        Test distance between line segments 2d for intersecting lines without a shared end point
        """

        line_a = np.array([np.array([0, 0]), np.array([5, 5])])
        line_b = np.array([np.array([0, 5]), np.array([5, 0])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(0.0, abs=1e-2)

    def test_distance_lineseg_to_lineseg_nd_intersect_point_on_line_2d(self):
        """
        Test distance between line segments 2d for intersecting lines with an end point of one line on the other line
        """

        line_a = np.array([np.array([2.5, 2.5]), np.array([5, 5])])
        line_b = np.array([np.array([0, 5]), np.array([5, 0])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(0.0, abs=1e-2)

    def test_distance_lineseg_to_lineseg_nd_skew_2d(self):
        """
        Test distance between line segments 2d for skew lines
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 5])])
        line_b = np.array([np.array([5, 5]), np.array([8, -10])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(5.0, rel=1e-3)

    def test_distance_lineseg_to_lineseg_nd_skew_2d_2(self):
        """
        Test gradient of the distance between line segments in 2d for skew lines
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 5])], dtype=float)
        line_b = np.array([np.array([5.0, 2.5]), np.array([10, 15])], dtype=float)

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(5.0)

    def test_distance_lineseg_to_lineseg_nd_skew_2d_3(self):
        """
        Test gradient of the distance between line segments in 2d for skew lines
        """

        line_a = np.array([np.array([5.0, 2.5]), np.array([10, 15])], dtype=float)
        line_b = np.array([np.array([0, 0]), np.array([0, 5])], dtype=float)

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(5.0)

    def test_distance_lineseg_to_lineseg_nd_parallel_grad_2d(self):
        """
        Test grad of distance between line segments 2d for parallel lines
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 5])], dtype=float)
        line_b = np.array([np.array([5, 0]), np.array([0, 15])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(
            test_result == np.array([0.0, 0.0])
        )  # goes to zero due to smooth_norm

    def test_distance_lineseg_to_lineseg_nd_shared_point_grad_2d(self):
        """
        Test distance between line segments 2d for intersecting lines with a shared point
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 5])], dtype=float)
        line_b = np.array([np.array([0, 0]), np.array([5, 5])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([0, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_intersect_grad_2d(self):
        """
        Test distance between line segments 2d for intersecting lines without a shared end point
        """

        line_a = np.array([np.array([0, 0]), np.array([5, 5])], dtype=float)
        line_b = np.array([np.array([0, 5]), np.array([5, 0])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([0, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_intersect_point_on_line_grad_2d(self):
        """
        Test distance between line segments 2d for intersecting lines with an end point of one line on the other line
        """

        line_a = np.array([np.array([2.5, 2.5]), np.array([5, 5])], dtype=float)
        line_b = np.array([np.array([0, 5]), np.array([5, 0])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([0, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_skew_grad_2d(self):
        """
        Test gradient of the distance between line segments in 2d for skew lines
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 5])], dtype=float)
        line_b = np.array([np.array([5.0, 2.5]), np.array([10, 15])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(
            test_result[0] == pytest.approx(np.array([-0.5, 0.0], dtype=float))
        )

        try:
            jax.test_util.check_grads(
                geo_utils.distance_lineseg_to_lineseg_nd,
                (line_a[0], line_a[1], line_b[0], line_b[1]),
                order=1,
            )
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )

    def test_distance_lineseg_to_lineseg_nd_2d_skew_grad(self):
        """
        Test distance between line segments 2d for skew lines
        """

        line_a = np.array([np.array([0, 0]), np.array([0, 10])], dtype=float)
        line_b = np.array([np.array([5, 0]), np.array([6, 15])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([-1, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_2d_skew_grad(self):
        """
        Test distance between line segments 2d for skew lines
        """

        line_b = np.array([np.array([0, 0]), np.array([0, 10])], dtype=float)
        line_a = np.array([np.array([5, 0]), np.array([6, 15])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([1, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_parallel(self):
        """
        Test distance between line segments 3d for parallel lines
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([0, 0, 5])])
        line_b = np.array([np.array([5, 0, 0]), np.array([5, 0, 5])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(
            5.0, rel=1e-1
        )  # inexact with parallel lines due to smooth max/min

    def test_distance_lineseg_to_lineseg_nd_shared_point(self):
        """
        Test distance between line segments 3d for intersecting lines with a shared point
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([0, 0, 5])])
        line_b = np.array([np.array([0, 0, 0]), np.array([5, 0, 5])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(0.0, abs=1e-2)

    def test_distance_lineseg_to_lineseg_nd_intersect(self):
        """
        Test distance between line segments 3d for intersecting lines without a shared end point
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([5, 5, 5])])
        line_b = np.array([np.array([0, 5, 0]), np.array([5, 0, 5])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(0.0, abs=1e-2)

    def test_distance_lineseg_to_lineseg_nd_intersect_point_on_line(self):
        """
        Test distance between line segments 3d for intersecting lines with an end point of one line on the other line
        """

        line_a = np.array([np.array([2.5, 2.5, 2.5]), np.array([5, 5, 5])])
        line_b = np.array([np.array([0, 5, 0]), np.array([5, 0, 5])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(0.0, abs=1e-2)

    def test_distance_lineseg_to_lineseg_nd_skew(self):
        """
        Test distance between line segments 3d for skew lines
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([0, 5, 5])])
        line_b = np.array([np.array([5, 5, 0]), np.array([5, 0, 5])])

        test_result = geo_utils.distance_lineseg_to_lineseg_nd(
            line_a_start=line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert test_result == pytest.approx(5.0, rel=1e-3)

    def test_distance_lineseg_to_lineseg_nd_parallel_grad(self):
        """
        Test grad of distance between line segments 3d for parallel lines
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([0, 0, 5])], dtype=float)
        line_b = np.array([np.array([5, 0, 0]), np.array([5, 0, 15])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        # the grad is prone to error in this region due to smooth max with many similar values,
        # just make sure grad is correct sign and reasonable magnitude.
        # Exact value should be [-1.0, 0.0, 0.0]
        assert np.all(test_result <= np.array([-0.5, 0.0, 0.0]))

    def test_distance_lineseg_to_lineseg_nd_almost_parallel_grad(self):
        """
        Test grad of distance between line segments 3d for parallel lines
        """

        line_a = np.array([np.array([0.01, 0, 0]), np.array([0, 0, 5])], dtype=float)
        line_b = np.array([np.array([5, 0, 0]), np.array([5, 0, 15])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([-1.0, 0.0, 0.0]))

    def test_distance_lineseg_to_lineseg_nd_shared_point_grad(self):
        """
        Test distance between line segments 3d for intersecting lines with a shared point
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([0, 0, 5])], dtype=float)
        line_b = np.array([np.array([0, 0, 0]), np.array([5, 0, 5])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([0, 0, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_intersect_grad(self):
        """
        Test distance between line segments 3d for intersecting lines without a shared end point
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([5, 5, 5])], dtype=float)
        line_b = np.array([np.array([0, 5, 0]), np.array([5, 0, 5])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([0, 0, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_intersect_point_on_line_grad(self):
        """
        Test distance between line segments 3d for intersecting lines with an end point of one line on the other line
        """

        line_a = np.array([np.array([2.5, 2.5, 2.5]), np.array([5, 5, 5])], dtype=float)
        line_b = np.array([np.array([0, 5, 0]), np.array([5, 0, 5])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([0, 0, 0], dtype=float))

    def test_distance_lineseg_to_lineseg_nd_skew_grad(self):
        """
        Test distance between line segments 3d for skew lines
        """

        line_a = np.array([np.array([0, 0, 0]), np.array([0, 5, 5])], dtype=float)
        line_b = np.array([np.array([5, 5, 0]), np.array([5, 0, 5])], dtype=float)

        test_result = self.distance_lineseg_to_lineseg_nd_grad(
            line_a[0],
            line_a_end=line_a[1],
            line_b_start=line_b[0],
            line_b_end=line_b[1],
        )

        assert np.all(test_result == np.array([-0.5, 0, 0], dtype=float))

        try:
            jax.test_util.check_grads(
                geo_utils.distance_lineseg_to_lineseg_nd,
                (line_a[0], line_a[1], line_b[0], line_b[1]),
                order=1,
            )
        except AssertionError:
            pytest.fail(
                "Unexpected AssertionError when checking gradients, gradients may be incorrect"
            )
