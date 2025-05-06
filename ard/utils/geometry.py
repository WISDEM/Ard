import numpy as np
import jax.numpy as jnp
import jax
from ard.utils.mathematics import smooth_min, smooth_norm


def _distance_lineseg_to_lineseg_coplanar(
    line_a_start: np.ndarray,
    line_a_end: np.ndarray,
    line_b_start: np.ndarray,
    line_b_end: np.ndarray,
) -> float:
    """Returns the distance between two finite line segments assuming the segments are coplanar.
    It is up to the user to check the required condition.

    Args:
        line_a_start (np.ndarray): start point of line a
        line_a_end (np.ndarray): end point of line a
        line_b_start (np.ndarray): start point of line b
        line_b_end (np.ndarray): end point of line b

    Returns:
        distance (float): the distance between the lines
    """

    # get distance between all pairs of end points
    a_start_to_b = distance_point_to_lineseg_nd(line_a_start, line_b_start, line_b_end)
    a_end_to_b = distance_point_to_lineseg_nd(line_a_end, line_b_start, line_b_end)
    b_start_to_a = distance_point_to_lineseg_nd(line_b_start, line_a_start, line_a_end)
    b_end_to_a = distance_point_to_lineseg_nd(line_b_end, line_a_start, line_a_end)

    distance = smooth_min(
        jnp.array([a_start_to_b, a_end_to_b, b_start_to_a, b_end_to_a])
    )

    return distance


_distance_lineseg_to_lineseg_coplanar = jax.jit(_distance_lineseg_to_lineseg_coplanar)


def distance_lineseg_to_lineseg_nd(
    line_a_start: np.ndarray,
    line_a_end: np.ndarray,
    line_b_start: np.ndarray,
    line_b_end: np.ndarray,
    tol=1e-12,
) -> float:
    """Find the distance between two line segments in 2d or 3d. This method is primarily based on reference [1].

    [1] Numerical Recipes: The Art of Scientific Computing by Press, et al. 3rd edition

    Args:
        line_a_start (np.ndarray): The start point of line segment "a" as either [x,y,z] or [x,y]
        line_a_end (np.ndarray): The end point of line segment "a" as either [x,y,z] or [x,y]
        line_b_start (np.ndarray): The start point of line segment "b" as either [x,y,z] or [x,y]
        line_b_end (np.ndarray): The end point of line segment "b" as either [x,y,z] or [x,y]
        tol (float, optional): If denominator in key equation is less than or equal tol, then an alternative method is used. Defaults to 0.0.

    Returns:
        float: Distance between the two line segments
    """

    def a_is_point(inputs0i) -> float:
        line_a_start = inputs0i[0]
        line_b_start = inputs0i[2]
        line_b_end = inputs0i[3]
        return distance_point_to_lineseg_nd(line_a_start, line_b_start, line_b_end)

    def b_is_point(inputs0i) -> float:
        line_a_start = inputs0i[0]
        line_a_end = inputs0i[1]
        line_b_start = inputs0i[2]
        return distance_point_to_lineseg_nd(line_b_start, line_a_start, line_a_end)

    def a_is_not_point(inputs0i) -> float:
        line_b_vector = inputs0i[5]
        return jax.lax.cond(
            jnp.all(line_b_vector == 0.0), b_is_point, a_and_b_are_lines, inputs0i
        )

    def a_and_b_are_lines(inputs0i) -> float:

        def denom_lt_tol(inputs1i) -> float:
            line_a_start = inputs1i[0]
            line_a_end = inputs1i[1]
            line_b_start = inputs1i[2]
            line_b_end = inputs1i[3]
            return _distance_lineseg_to_lineseg_coplanar(
                line_a_start=line_a_start,
                line_a_end=line_a_end,
                line_b_start=line_b_start,
                line_b_end=line_b_end,
            )

        def denom_gt_tol(inputs1i) -> float:
            line_a_start = inputs1i[0]
            line_a_end = inputs1i[1]
            line_b_start = inputs1i[2]
            line_b_end = inputs1i[3]
            line_a_vector = inputs1i[4]
            line_b_vector = inputs1i[5]
            denominator = inputs1i[6]

            a = line_a_start
            v = line_a_vector
            x = line_b_start
            u = line_b_vector

            s_numerator = jnp.linalg.det(jnp.array([a - x, u, jnp.cross(u, v)]).T)
            t_numerator = jnp.linalg.det(jnp.array([a - x, v, jnp.cross(u, v)]).T)

            s = s_numerator / denominator
            t = t_numerator / denominator

            # Get closest point along the lines
            # if s or t > 1, use end point of line
            def st_gt_1(inputs23i) -> np.ndarray:
                line_end = inputs23i[1]
                return jnp.array(line_end, dtype=float)

            # if s or t < 0, use start point of line
            def st_lt_0(inputs23i) -> np.ndarray:
                line_start = inputs23i[0]
                return jnp.array(line_start, dtype=float)

            # otherwise compute the closest point on line using the parametric form of the line segment
            def st_gt_0_lt_1(inputs23i) -> np.ndarray:
                line_start = inputs23i[0]
                line_vector = inputs23i[2]
                st = inputs23i[3]
                return jnp.array(line_start + st * line_vector, dtype=float)

            def st_lt_1(inputs23i) -> np.ndarray:
                st = inputs23i[3]
                return jax.lax.cond(st < 0, st_lt_0, st_gt_0_lt_1, inputs23i)

            # get closest point on lines a and b to each other
            inputs2o = [line_a_start, line_a_end, line_a_vector, s]
            closest_point_line_a = jax.lax.cond(s > 1, st_gt_1, st_lt_1, inputs2o)
            inputs3o = [line_b_start, line_b_end, line_b_vector, t]
            closest_point_line_b = jax.lax.cond(t > 1, st_gt_1, st_lt_1, inputs3o)

            # the distance between the line segments is the distance between the closest points (in many cases)
            parametric_distance = smooth_norm(
                closest_point_line_b - closest_point_line_a
            )

            # parametric approach can miss cases, so compare with point to line distances
            distance_point_a_line_b = distance_point_to_lineseg_nd(
                closest_point_line_a, line_b_start, line_b_end
            )
            distance_point_b_line_a = distance_point_to_lineseg_nd(
                closest_point_line_b, line_a_start, line_a_end
            )
            distance = smooth_min(
                jnp.array(
                    [
                        parametric_distance,
                        distance_point_a_line_b,
                        distance_point_b_line_a,
                    ]
                )
            )

            return distance

        line_a_start = inputs0i[0]
        line_a_end = inputs0i[1]
        line_b_start = inputs0i[2]
        line_b_end = inputs0i[3]
        line_a_vector = inputs0i[4]
        line_b_vector = inputs0i[5]

        # find s and t (point along segment where the segments are closest to each other) using eq. 21.4.17 in [1]
        denominator = smooth_norm(jnp.cross(line_b_vector, line_a_vector)) ** 2

        inputs1o = [
            line_a_start,
            line_a_end,
            line_b_start,
            line_b_end,
            line_a_vector,
            line_b_vector,
            denominator,
        ]

        distance = jax.lax.cond(
            denominator <= tol, denom_lt_tol, denom_gt_tol, inputs1o
        )

        return distance

    # if 2d given, then pad with zeros to get 3d points
    pad_width = len(line_a_start)
    line_a_start = jnp.pad(line_a_start, (0, 3 - pad_width))
    line_a_end = jnp.pad(line_a_end, (0, 3 - pad_width))
    line_b_start = jnp.pad(line_b_start, (0, 3 - pad_width))
    line_b_end = jnp.pad(line_b_end, (0, 3 - pad_width))

    line_a_vector = line_a_end - line_a_start
    line_b_vector = line_b_end - line_b_start

    inputs0o = [
        line_a_start,
        line_a_end,
        line_b_start,
        line_b_end,
        line_a_vector,
        line_b_vector,
    ]

    distance = jax.lax.cond(
        jnp.all(line_a_vector == 0.0), a_is_point, a_is_not_point, inputs0o
    )

    return distance


distance_lineseg_to_lineseg_nd = jax.jit(distance_lineseg_to_lineseg_nd)


def distance_point_to_lineseg_nd(
    point: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray
) -> float:
    """Find the distance from a point to a finite line segment in N-Dimensions

    Args:
        point (np.ndarray): point of interest [x,y,...]
        segment_start (np.ndarray): start point of line segment [x,y,...]
        segment_end (np.ndarray): end point of line segment [x,y,...]

    Returns:
        distance (float): shortest distance between the point and finite line segment
    """

    def if_point_to_point(inputs) -> float:
        point = inputs[0]
        segment_start = inputs[1]
        return smooth_norm(jnp.subtract(segment_start, point))

    def if_point_to_line_seg(inputs) -> float:
        point = inputs[0]
        segment_start = inputs[1]
        segment_end = inputs[2]
        segment_vector = inputs[3]

        # get the closest point on the line segment to the point of interest
        closest_point = get_closest_point(
            point, segment_start, segment_end, segment_vector
        )

        # the distance from the point to the line is the distance from the point to the closest point on the line
        return smooth_norm(jnp.subtract(point, closest_point))

    # get the vector of the line segment
    segment_vector = segment_end - segment_start

    # if the segment is a point, then get the distance to that point
    distance = jax.lax.cond(
        jnp.all(segment_vector == 0),
        if_point_to_point,
        if_point_to_line_seg,
        [point, segment_start, segment_end, segment_vector],
    )

    return distance


distance_point_to_lineseg_nd = jax.jit(distance_point_to_lineseg_nd)


def get_closest_point(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    segment_vector: np.ndarray,
) -> np.ndarray:
    """Get the closest point on the line segment to the point of interest in N-Dimensions

    Args:
        point (np.ndarray): point of interest [x,y,...]
        segment_start (np.ndarray): start point of line segment [x,y,...]
        segment_end (np.ndarray): end point of line segment [x,y,...]
        segment_vector (np.ndarray): segment_end - segment_start

    Returns:
        np.ndarray: closest point on the line segment to the point of interest
    """

    # calculate the distance to the starting point
    start_to_point_vector = point - segment_start

    # calculate the unit vector projection of the start to point vector on the line segment
    projection = jnp.divide(
        jnp.dot(start_to_point_vector, segment_vector),
        jnp.dot(segment_vector, segment_vector),
    )

    def lt_0(inputs) -> np.ndarray:
        segment_start = inputs[1]
        return jnp.array(segment_start, dtype=float)

    def gt_1(inputs) -> np.ndarray:
        segment_end = inputs[2]
        return jnp.array(segment_end, dtype=float)

    def gt_0(inputs) -> np.ndarray:
        projection = inputs[0]
        return jax.lax.cond(projection > 1, gt_1, lt_1_gt_0, inputs)

    def lt_1_gt_0(inputs) -> np.ndarray:
        projection = inputs[0]
        segment_start = inputs[1]
        segment_vector = inputs[3]
        return jnp.array(segment_start + projection * segment_vector, dtype=float)

    return jax.lax.cond(
        projection < 0,
        lt_0,
        gt_0,
        [projection, segment_start, segment_end, segment_vector],
    )


get_closest_point = jax.jit(get_closest_point)
