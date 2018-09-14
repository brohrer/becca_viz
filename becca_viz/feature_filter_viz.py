"""
Show what's going on inside an input filter.
"""
import numpy as np
import matplotlib.pyplot as plt

import becca_viz.viz_tools as vt


def render(
    filter,
    bbox,
    y_pool,
    pool_viz_map,
    radius=0,
    n_commands=None,
    n_inputs=None,
    n_bundles_all=None,
):
    """
    Make a picture of the discretization that happens in an input filter.

    To simplify the notation on transforms here, the following substitutions
    will be used:
        A:  feature pool in visualized order
        B:  feature pool in natural order (the order passed in)
        C:  features in natural order
        D:  features in visualized order

    Parameters
    ----------
    bbox: list of floats
        Of the form [x_bottom, x_top, y_left, y_right]
    filter: InputFilter
    y_pool: array of floats
    pool_viz_map: 2D array of ints
    radius: float
    n_commands, n_inputs: int
    n_bundles_all: list of int

    Returns
    -------
    y_features: array of floats
    feature_viz_map: 2D array of ints
    """
    xmin, xmax, ymin, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin

    n_pool = y_pool.size
    y_pool_spacing = (height - 2 * radius) / (n_pool + 1)
    y_A = np.linspace(
        ymin + radius + y_pool_spacing,
        ymax - radius - y_pool_spacing,
        num=n_pool,
        endpoint=True,
    )

    # Place labels alongside the features to show where they come from.
    if (
        n_commands is not None
        and n_inputs is not None
        and n_bundles_all is not None
    ):
        i_last = n_commands - 1
        ylabel = "commands"
        text_sep = width * vt.text_shift
        plt.text(
            xmin - text_sep,
            y_A[i_last],
            ylabel,
            fontsize=4,
            color=vt.copper,
            verticalalignment="top",
            horizontalalignment="right",
            rotation=-90,
            family="sans-serif",
        )
        i_last += n_inputs
        ylabel = "inputs"
        text_sep = width * vt.text_shift
        plt.text(
            xmin - text_sep,
            y_A[i_last],
            ylabel,
            fontsize=4,
            color=vt.copper,
            verticalalignment="top",
            horizontalalignment="right",
            rotation=-90,
            family="sans-serif",
        )

        for i_ziptie, n_bundles in enumerate(n_bundles_all):
            if n_bundles > 0:
                i_last += n_bundles
                ylabel = "ziptie " + str(i_ziptie)
                text_sep = width * vt.text_shift
                plt.text(
                    xmin - text_sep,
                    y_A[i_last],
                    ylabel,
                    fontsize=4,
                    color=vt.copper,
                    verticalalignment="top",
                    horizontalalignment="right",
                    rotation=-90,
                    family="sans-serif",
                )

    map_AB = pool_viz_map.T
    activities_B = filter.candidate_activities
    activities_A = np.matmul(map_AB, activities_B)

    n_inputs = (np.where(np.sum(filter.mapping, axis=0))[0]).size
    map_BC = filter.mapping[:n_pool, :n_inputs]
    map_AC = np.matmul(map_AB, map_BC)
    order_CD = np.argsort(np.argsort(np.matmul(y_A, map_AC)))
    map_CD = np.zeros((n_inputs, n_inputs), dtype=np.int)
    map_CD[np.arange(n_inputs, dtype=np.int), order_CD] = 1
    map_AD = np.matmul(map_AC, map_CD)

    y_spacing = (height - 2 * radius) / (n_inputs + 1)
    y_D = np.linspace(
        ymin + radius + y_spacing,
        ymax - radius - y_spacing,
        num=n_inputs,
        endpoint=True,
    )
    activities_D = np.matmul(activities_A, map_AD)

    i_DA = np.matmul(np.arange(n_pool, dtype=np.int), map_AD)
    # Show the filter's selection
    for i_D, activity in enumerate(activities_D):
        i_A = i_DA[i_D]
        y_end = y_D[i_D]
        y_start = y_A[i_A]
        vt.plot_curve_activity_horiz(y_start, y_end, xmin, xmax, activity)
        vt.plot_point_activity(xmax, y_end, activity, y_spacing)

    return y_D, map_CD
