"""
Show what's going on inside an input filter.
"""
import numpy as np

import becca_viz.viz_tools as vt


def render(filter, bbox, x_pool_prev, pool_viz_map, y_prev,  radius=0):
    """
    Make a picture of the discretization that happens in an input filter.

    To simplify the notation on transforms here, the following substitutions
    will be used:
        A:  candidate pool in visualized order
        B:  candidate pool in natural order (the order passed in)
        C:  inputs in natirual order
        D:  inputs in visualized order

    Parameters
    ----------
    bbox: list of floats
        Of the form [x_bottom, x_top, y_left, y_right]
    filter: InputFilter
    radius: float

    Returns
    -------
    x_cables: array of floats
        The absolute x positions of the cables.
    i_to_viz_cables: 2D array of ints
        Mapping from cable pool to their visualization order
    """
    xmin, xmax, ymin, ymax = bbox
    frame_width = xmax - xmin
    # frame_height = ymax - ymin

    n_pool = x_pool_prev.size
    x_pool_spacing = (frame_width - 2 * radius) / (x_pool_prev.size + 1)
    x_A = np.linspace(
        xmin + radius + x_pool_spacing,
        xmax - radius - x_pool_spacing,
        num=n_pool,
        endpoint=True,
    )

    map_BA = pool_viz_map
    map_AB = map_BA.T
    activities_B = filter.candidate_activities
    activities_A = np.matmul(activities_B, map_BA)

    # Connect the previous block(s) to this one.
    for i_A, activity in enumerate(activities_A):
        vt.plot_curve_activity(
            x_pool_prev[i_A], x_A[i_A],
            y_prev, ymin, activity)
        vt.plot_point_activity(
            x_A[i_A], ymin, activity, x_pool_spacing)

    n_inputs = (np.where(np.sum(filter.mapping, axis=0))[0]).size
    map_BC = filter.mapping[:n_pool, :n_inputs]
    map_AC = np.matmul(map_AB, map_BC)
    order_CD = np.argsort(np.argsort(np.matmul(x_A, map_AC)))
    map_CD = np.zeros((n_inputs, n_inputs), dtype=np.int)
    map_CD[np.arange(n_inputs, dtype=np.int), order_CD] = 1
    map_AD = np.matmul(map_AC, map_CD)

    x_spacing = (frame_width - 2 * radius) / (n_inputs + 1)
    x_D = np.linspace(
        xmin + radius + x_spacing,
        xmax - radius - x_spacing,
        num=n_inputs,
        endpoint=True,
    )
    activities_D = np.matmul(activities_A, map_AD)

    i_DA = np.matmul(np.arange(n_pool, dtype=np.int), map_AD)
    # Show the filter's selection
    for i_D, activity in enumerate(activities_D):
        i_A = i_DA[i_D]
        x_end = x_D[i_D]
        x_start = x_A[i_A]
        vt.plot_curve_activity(x_start, x_end, ymin, ymax, activity)
        vt.plot_point_activity(x_end, ymax, activity, x_spacing)

    return x_D, map_CD
