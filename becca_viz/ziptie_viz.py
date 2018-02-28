import os

import matplotlib.pyplot as plt
import numpy as np

import becca_viz.viz_tools as vt


def render(ziptie, bbox, x_inputs, cable_viz_map, y_prev, radius=0):
    """
    Make a picture of the cable-to-bundle combinations that happen
    in a ziptie.

    Parameters
    ----------
    bbox: list of floats
        Of the form [x_bottom, x_top, y_left, y_right]
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
    frame_height = ymax - ymin
    n_inputs = x_inputs.size

    x_cable_spacing = (frame_width - 2 * radius) / (n_inputs + 1)
    x_cable = np.arange(
        xmin + radius + x_cable_spacing, 
        xmax - radius,
        x_cable_spacing)

    cable_activities = ziptie.cable_activities[:n_inputs]
    # Connect the previous block(s) to this one.
    i_to_viz_cables = np.where(cable_viz_map)[1]
    for i_cable, activity in enumerate(cable_activities):
        i_cable_viz = i_to_viz_cables[i_cable]
        vt.plot_curve_activity(
            x_inputs[i_cable_viz],
            x_cable[i_cable_viz],
            y_prev, ymin,
            activity)
        vt.plot_point_activity(
            x_cable[i_cable_viz], ymin, activity, x_cable_spacing)

    n_bundles = ziptie.n_bundles
    if n_bundles > 0:
        x_bundle_spacing = (frame_width - 2 * radius) / (n_bundles + 1)
        x_bundles = np.arange(
            xmin + radius + x_bundle_spacing, 
            xmax - radius,
            x_bundle_spacing)
        
        bundle_viz_map = np.zeros((n_bundles, n_bundles), dtype=np.int)

        bundle_score = np.mean(
            np.matmul(x_cable, cable_viz_map.T)[:, np.newaxis] *
            ziptie.mapping[:, :n_bundles], axis=0)
        bundle_viz_map[np.arange(n_bundles), np.argsort(bundle_score)] = 1
        viz_to_i_bundle = np.where(bundle_viz_map)[1]
        x_bundle_viz = np.matmul(x_bundles, bundle_viz_map)

        for i_cable, activity in enumerate(cable_activities):
            i_cable_viz = i_to_viz_cables[i_cable]
            for i_bundle in np.where(
                    ziptie.mapping[i_cable,:])[0]:
                i_bundle_viz = viz_to_i_bundle[i_bundle]
                x_end = x_bundles[i_bundle_viz]
                x_start = x_cable[i_cable_viz]
                vt.plot_curve_activity(x_start, x_end, ymin, ymax, activity)

        for i_bundle, activity in enumerate(ziptie.bundle_activities):
            x_end = x_bundle_viz[i_bundle]
            vt.plot_point_activity(x_end, ymax, activity, x_bundle_spacing)
    else:
        x_bundles = None
        bundle_viz_map = None

    return x_bundles, bundle_viz_map
