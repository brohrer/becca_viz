import numpy as np

import becca_viz.viz_tools as vt


def render(featurizer, bbox, viz_maps, radius=0):
    """

    Parameters
    ----------
    featurizer : Featurizer
    bbox: list of floats
        In the format [xmin, xmax, ymin, ymax]
    viz_maps: list of 2D arrays of ints
        Maps between cable candidate pools and their visualization order.
    radius: float

    Returns
    -------
    y_pool_feature: array of floats
        The absolute positions of each member of the feature pool.
    feature_viz_map: 2D array of floats
        Map between the feature pool and their visualization order.
    """
    xmin, xmax, ymin, ymax = bbox
    # frame_width = xmax - xmin
    frame_height = ymax - ymin

    if viz_maps[-1] is None:
        viz_maps = viz_maps[:-1]
    block_rows = []
    for i_map in np.arange(len(viz_maps), dtype=np.int):
        block_row = []
        mrows, mcols = viz_maps[i_map].shape
        for j_map in np.arange(len(viz_maps), dtype=np.int):
            nrows, ncols = viz_maps[j_map].shape
            if i_map == j_map:
                block_row.append(np.fliplr(viz_maps[i_map]))
            else:
                block_row.append(np.zeros((mrows, ncols), dtype=np.int))
        block_rows.append(block_row)

    map_AB = np.block(block_rows).T
    map_BC = featurizer.mapping
    order_CD = np.argsort(np.argsort(np.matmul(
        np.arange(map_AB.shape[0]),
        np.matmul(map_AB, map_BC))))
    n_D = map_BC.shape[1]
    map_CD = np.zeros((n_D, n_D), dtype=np.int)
    map_CD[np.arange(n_D, dtype=np.int), order_CD] = 1

    activities_B = []
    for level_activities in featurizer.activities:
        activities_B += list(level_activities)
    activities_B = np.array(activities_B)
    activities_D = np.matmul(activities_B, np.matmul(map_BC, map_CD))

    y_spacing = (frame_height - 2 * radius) / (n_D + 1)
    y_D = np.linspace(
        ymin + radius + y_spacing,
        ymax - radius - y_spacing,
        num=n_D,
        endpoint=True,
    )

    for i_D, activity in enumerate(activities_D):
        # vt.plot_curve_activity_horiz(
        #     y_start, y_end, xmin, xmax, activity, start=.3)
        vt.plot_point_activity(xmin, y_D[i_D], activity, y_spacing)

    return y_D, map_CD
