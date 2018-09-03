# import os

# import matplotlib.pyplot as plt
import numpy as np

import becca_viz.viz_tools as vt


def render(ziptie, bbox, x_inputs, cable_viz_map, y_prev, radius=0):
    """
    Make a picture of the cable-to-bundle combinations that happen
    in a ziptie.

    To keep notation simpler, maps will be referred to as map_xy,
    where x and y can be any of the following:
        A:  cable visualization order
        B:  cable natural order (in which the ziptie receives it)
        C:  bundle natural order (in which ziptie.map creates it)
        D:  bundle visualization order

    Parameters
    ----------
    bbox: list of floats
        Of the form [x_left, x_right, y_bottom, y_top]
    ziptie: Ziptie
    cable_viz_map: 2D array of ints
    y_prev: float
    radius: float

    Returns
    -------
    x_bundles: array of floats
        The absolute x positions of the cables.
    bundle_viz_map: 2D array of ints
        The indices for ordering bundles to align with
        the visualization.
    """
    xmin, xmax, ymin, ymax = bbox
    frame_width = xmax - xmin
    # frame_height = ymax - ymin
    n_A = x_inputs.size
    n_D = ziptie.n_bundles

    x_A_spacing = (frame_width - 2 * radius) / (n_A + 1)
    x_A = np.linspace(
        xmin + radius + x_A_spacing,
        xmax - radius - x_A_spacing,
        num=n_A,
        endpoint=True,
    )

    map_BA = cable_viz_map
    map_AB = map_BA.T
    activities_B = ziptie.cable_activities[:n_A]
    # activities_A = np.matmul(map_AB, activities_B)
    # activities_A = np.matmul(activities_B, map_BA)

    if n_D > 0:
        x_D_spacing = (frame_width - 2 * radius) / (n_D + 1)
        x_D = np.linspace(
            xmin + radius + x_D_spacing,
            xmax - radius - x_D_spacing,
            num=n_D,
            endpoint=True,
        )

        map_BC = ziptie.mapping[:n_A, :n_D]
        x_B = np.matmul(x_A, map_AB)
        bundle_score = (np.sum(x_B[:, np.newaxis] * map_BC, axis=0) /
                        np.sum(map_BC, axis=0))
        map_CD = np.zeros((n_D, n_D), dtype=np.int)
        map_CD[np.arange(n_D, dtype=np.int),
               np.argsort(np.argsort(bundle_score))] = 1
        i_CD = np.matmul(map_CD, np.arange(n_D, dtype=np.int))
        i_BA = np.matmul(np.arange(n_A, dtype=np.int), map_AB)

        for i_B, activity in enumerate(activities_B):
            i_A = i_BA[i_B]
            for i_C in np.where(map_BC[i_B, :])[0]:
                i_D = i_CD[i_C]
                x_end = x_D[i_D]
                x_start = x_A[i_A]
                vt.plot_curve_activity(x_start, x_end, ymin, ymax, activity)
        activities_C = ziptie.bundle_activities
        activities_D = np.matmul(activities_C, map_CD)
        for i_D, activity in enumerate(activities_D):
            vt.plot_point_activity(x_D[i_D], ymax, activity, x_D_spacing)
    else:
        x_D = None
        map_CD = None

    x_bundles = x_D
    bundle_viz_map = map_CD
    return x_bundles, bundle_viz_map
