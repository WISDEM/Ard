import numpy as np
import jax.numpy as jnp
import jax
from ard.utils.mathematics import smooth_max, smooth_min, smooth_norm


# TODO test
def distance_multi_point_to_multi_polygon_ray_casting(
    boundary_vertices,
    boundary_normals,
    points_x,
    points_y,
    discrete=False,
    s=700,
    tol=1e-6,
    return_region=False,
    regions=None,
    c=None,
):
    """
    Calculate the distance from each point to the nearest point on a polygon or set of polygons using
    the ray-casting algorithm. Negative means the turbine is inside the boundary.
    This implementation is based on FLOWFarm.jl (https://github.com/byuflowlab/FLOWFarm.jl)

    Args:
        boundary_vertices (np.ndarray or list of np.ndarray): Vertices of the boundary in
            counterclockwise order. If `discrete` is True, this should be a list of arrays.
        boundary_normals (np.ndarray or list of np.ndarray): Unit normal vectors for each
            boundary face in counterclockwise order. If `discrete` is True, this should be
            a list of arrays.
        points_x (np.ndarray): points x coordinates.
        points_y (np.ndarray): points y coordinates.
        discrete (bool, optional): If True, indicates the boundary is made of multiple discrete regions.
            Defaults to False.
        s (float, optional): Smoothing factor for smooth max. Defaults to 700.
        tol (float, optional): Tolerance for determining proximity. Defaults to 1e-6.
        return_region (bool, optional): If True, return a vector specifying which region each turbine is in.
            Defaults to False.
        regions (list, optional): Predefined region assignments for turbines. Defaults to None.
        c (np.ndarray, optional): Preallocated array for constraint values. Defaults to None.

    Returns:
        np.ndarray: Constraint values for each turbine.
        np.ndarray (optional): Region assignments for each turbine (if `return_region` is True).
    """
    # Number of turbines
    n_points = len(points_x)

    # Initialize constraint output values
    if c is None:
        c = np.zeros(n_points, dtype=float)

    # Single region
    if not discrete:
        for i in range(n_points):
            # Determine if the point is contained in the polygon
            c[i] = distance_point_to_polygon_ray_casting(
                np.array([points_x[i], points_y[i]]),
                boundary_vertices,
                boundary_normals,
                s=s,
                shift=tol,
            )
        return c

    # Multiple discrete regions with predefined region assignments
    elif regions is not None and len(regions) > 0:
        for i in range(n_points):
            # Determine if the point is contained in the assigned polygonal region
            c[i] = distance_point_to_polygon_ray_casting(
                np.array([points_x[i], points_y[i]]),
                boundary_vertices[regions[i]],
                boundary_normals[regions[i]],
                s=s,
                shift=tol,
            )
        return c

    # Multiple discrete regions without predefined region assignments
    else:
        # Number of regions
        nregions = len(boundary_vertices)

        # Initialize region and status arrays
        region = np.zeros(n_points, dtype=int)
        status = np.zeros(n_points, dtype=int)

        # Initialize array to hold distances from each turbine to the closest boundary face
        turbine_to_region_distance = np.zeros(nregions, dtype=float)

        for i in range(n_points):
            # Iterate through each region
            for k in range(nregions):
                # Check if the point is in this region
                ctmp = distance_point_to_polygon_ray_casting(
                    np.array([points_x[i], points_y[i]]),
                    boundary_vertices[k],
                    boundary_normals[k],
                    s=s,
                    shift=tol,
                    return_distance=False,
                )
                if ctmp <= 0:  # Negative if in boundary
                    c[i] = distance_point_to_polygon_ray_casting(
                        np.array([points_x[i], points_y[i]]),
                        boundary_vertices[k],
                        boundary_normals[k],
                        s=s,
                        shift=tol,
                        return_distance=True,
                    )
                    status[i] = 1
                    if return_region:
                        region[i] = k
                    break

            # Check if the turbine is in none of the regions
            if status[i] == 0:
                for k in range(nregions):
                    # Calculate distance to each region
                    turbine_to_region_distance[k] = (
                        distance_point_to_polygon_ray_casting(
                            np.array([points_x[i], points_y[i]]),
                            boundary_vertices[k],
                            boundary_normals[k],
                            s=s,
                            shift=tol,
                            return_distance=True,
                        )
                    )

                # Magnitude of the constraint value
                c[i] = -smooth_max(-turbine_to_region_distance, s=s)

                # Set status to indicate that the turbine has been assigned
                status[i] = 1

                # Indicate closest region
                region[i] = np.argmin(turbine_to_region_distance)

        if return_region:
            return c, region
        return c


def distance_point_to_polygon_ray_casting(
    point: jnp.ndarray,
    vertices: jnp.ndarray,
    s: float = 700,
    shift: float = 1e-10,
    return_distance: bool = True,
):
    """
    Determine the signed distance from a point to a polygon in a differentiable way.

    Args:
        point (jnp.ndarray): Point of interest (2D vector).
        vertices (jnp.ndarray): Vertices of the polygon (Nx2 array).
        s (float, optional): Smoothing factor for the smoothmin function. Defaults to 700.
        shift (float, optional): Small shift to handle edge cases. Defaults to 1e-10.
        return_distance (bool, optional): Whether to return the signed distance or just
            inside/outside status. Defaults to True.

    Returns:
        float: Signed distance or inside/outside status.
    """
    # Ensure inputs are JAX arrays with explicit data types
    point = jnp.asarray(point, dtype=jnp.float32)
    vertices = jnp.asarray(vertices, dtype=jnp.float32)

    # Add the first vertex to the end to close the polygon loop
    vertices = jnp.vstack([vertices, vertices[0]])

    # Define a function to process a single edge
    def process_edge(edge_start, edge_end, point):
        # Check if the x-coordinate of the point is between the x-coordinates of the edge
        x_condition = ((edge_start[0] <= point[0]) & (point[0] < edge_end[0])) | (
            (edge_start[0] >= point[0]) & (point[0] > edge_end[0])
        )

        # Calculate the y-coordinate of the edge at the x-coordinate of the point
        y = (edge_end[1] - edge_start[1]) / (edge_end[0] - edge_start[0] + shift) * (
            point[0] - edge_start[0]
        ) + edge_start[1]

        # Determine if the point is below the edge
        is_below = x_condition & (point[1] < y)

        # Calculate the distance to the edge
        distance = distance_point_to_lineseg_nd(point, edge_start, edge_end)

        return is_below, distance

    # Vectorize the edge processing function
    edge_starts = vertices[:-1]
    edge_ends = vertices[1:]
    is_below, distances = jax.vmap(process_edge, in_axes=(0, 0, None))(
        edge_starts, edge_ends, point
    )

    # Count the number of intersections
    intersection_counter = jnp.sum(is_below)

    # Compute the signed distance
    if return_distance:
        c = smooth_min(distances, s=s)
        c = jax.lax.cond(
            intersection_counter % 2 == 1,
            lambda _: -c,
            lambda _: c,
            operand=None,
        )
    else:
        c = jax.lax.cond(
            intersection_counter % 2 == 1,
            lambda _: -1.0,
            lambda _: 1.0,
            operand=None,
        )
    return c


distance_point_to_polygon_ray_casting = jax.jit(distance_point_to_polygon_ray_casting)


def polygon_normals_calculator(
    boundary_vertices: np.ndarray, n_polygons: int = 1
) -> list[np.ndarray]:
    """
    Calculate unit vectors perpendicular to each edge of each polygon in a set of polygons.
    This implementation is based on FLOWFarm.jl (https://github.com/byuflowlab/FLOWFarm.jl).
    This function is not intended to be differentiable wrt n_polygons.

    Args:
        boundary_vertices (list of np.ndarray): List of m-by-2 arrays, where each array contains
            the boundary vertices of a polygon in counterclockwise order.
        nboundaries (int, optional): The number of boundaries in the set. Defaults to 1.

    Returns:
        list of np.ndarray: List of m-by-2 arrays of unit vectors perpendicular to each edge
            of each polygon.
    """
    if n_polygons == 1:
        # Single polygon case
        return single_polygon_normals_calculator(boundary_vertices)
    else:
        # Multiple polygons case
        return multi_polygon_normals_calculator(boundary_vertices)


def multi_polygon_normals_calculator(
    boundary_vertices: list[np.ndarray],
) -> jnp.ndarray:
    """
    Calculate unit vectors perpendicular to each edge of each polygon in a set of polygons.
    This implementation is based on FLOWFarm.jl (https://github.com/byuflowlab/FLOWFarm.jl).

    Args:
        boundary_vertices (list of np.ndarray): List of m-by-2 arrays, where each array contains
            the boundary vertices of a polygon in counterclockwise order.
        nboundaries (int, optional): The number of boundaries in the set. Defaults to 1.

    Returns:
        list of np.ndarray: List of m-by-2 arrays of unit vectors perpendicular to each edge
            of each polygon.
    """
    return [
        single_polygon_normals_calculator(vertices) for vertices in boundary_vertices
    ]


multi_polygon_normals_calculator = jax.jit(multi_polygon_normals_calculator)


def single_polygon_normals_calculator(boundary_vertices: np.ndarray) -> jnp.ndarray:
    """
    Calculate unit vectors perpendicular to each edge of a polygon pointing into the polygon.
    This implementation is based on FLOWFarm.jl (https://github.com/byuflowlab/FLOWFarm.jl).

    Args:
        boundary_vertices (np.ndarray): m-by-2 array containing all the boundary vertices
            in counterclockwise order.

    Returns:
        np.ndarray: m-by-2 array of unit vectors perpendicular to each edge of the polygon pointing into the polygon.
    """

    # Add the first vertex to the end to form a closed loop
    boundary_vertices = jnp.vstack([boundary_vertices, boundary_vertices[0]])

    # Compute the differences (dx, dy) for each edge
    dx = boundary_vertices[1:, 0] - boundary_vertices[:-1, 0]
    dy = boundary_vertices[1:, 1] - boundary_vertices[:-1, 1]

    # Create vectors normal to the boundary edges
    boundary_normals = jnp.stack([-dy, dx], axis=1)

    # Normalize the vectors
    norms = jnp.linalg.norm(boundary_normals, axis=1, keepdims=True)
    boundary_normals = boundary_normals / norms

    return [boundary_normals]


single_polygon_normals_calculator = jax.jit(single_polygon_normals_calculator)


def point_on_line(p: np.ndarray, v1: np.ndarray, v2: np.ndarray, tol=1e-6):
    """
    Determine if a point lies on a line segment.

    Given a line determined by two points (v1 and v2), determine if the point (p) lies on the line
    between those points within a given tolerance.

    Args:
        p (np.ndarray): Point of interest (2D vector).
        v1 (np.ndarray): First vertex of the line (2D vector).
        v2 (np.ndarray): Second vertex of the line (2D vector).
        tol (float): Tolerance for determining co-linearity.

    Returns:
        bool: True if the point lies on the line, False otherwise.
    """

    d = distance_point_to_lineseg_nd(p, v1, v2)

    return jnp.isclose(d, 0.0, atol=tol)


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
        return jnp.float64(smooth_norm(segment_start - point))

    def if_point_to_line_seg(inputs) -> float:
        point = inputs[0]
        segment_start = inputs[1]
        segment_end = inputs[2]
        segment_vector = inputs[3]

        # get the closest point on the line segment to the point of interest
        closest_point = get_closest_point_on_line_seg(
            point, segment_start, segment_end, segment_vector
        )

        # the distance from the point to the line is the distance from the point to the closest point on the line
        return jnp.float64(smooth_norm(point - closest_point))

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


def get_closest_point_on_line_seg(
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
        return jnp.array(segment_start, dtype=jnp.float64)

    def gt_1(inputs) -> np.ndarray:
        segment_end = inputs[2]
        return jnp.array(segment_end, dtype=jnp.float64)

    def gt_0(inputs) -> np.ndarray:
        projection = inputs[0]
        return jax.lax.cond(projection > 1, gt_1, lt_1_gt_0, inputs)

    def lt_1_gt_0(inputs) -> np.ndarray:
        projection = inputs[0]
        segment_start = inputs[1]
        segment_vector = inputs[3]
        return jnp.array(segment_start + projection * segment_vector, dtype=jnp.float64)

    return jax.lax.cond(
        projection < 0,
        lt_0,
        gt_0,
        [projection, segment_start, segment_end, segment_vector],
    )


get_closest_point_on_line_seg = jax.jit(get_closest_point_on_line_seg)
