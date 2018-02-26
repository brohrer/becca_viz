"""
Show what's going on inside a ziptie.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import becca_viz.viz_tools as vt


def render(ziptie, bbox, x_inputs, cable_viz_map, y_prev,  radius=0):
    """
    Make a picture of the cable-to-bundle combinations that happen
    in a ziptie.

    Parameters
    ----------
    bbox: list of floats
        Of the form [x_bottom, x_top, y_left, y_right]
    ziptie: Ziptie
    radius: float

    Returns
    -------
    x_inputs: array of floats
        The absolute x positions of the cables.
    i_to_viz_cables: array of ints
        The indices for ordering cables to align with
        the visualization.
    """

    xmin, xmax, ymin, ymax = bbox
    frame_width = xmax - xmin
    frame_height = ymax - ymin

    x_cable_spacing = (frame_width - 2 * radius) / (x_inputs.size + 1)
    x_cable = np.arange(
        xmin + radius + x_cable_spacing, 
        xmax - radius,
        x_cable_spacing)

    # Connect the previous block(s) to this one.
    i_to_viz_cables = np.where(cable_viz_map)[1]
    # ordered_activities = ziptie.cable_activities[i_to_viz_cables]
    # for i_act, activity in enumerate(ordered_activities):
    for i_cable, activity in enumerate(ziptie.cable_activities[i_in_use]):
        
        vt.plot_curve_activity(
            x_inputs[i_act], x_cable[i_act], y_prev, ymin, activity)
        vt.plot_point_activity(
            x_cable[i_act], ymin, activity, x_cable_spacing)

    x_bundle_spacing = (frame_width - 2 * radius) / (ziptie.n_bundles + 1)
    x_bundles = np.arange(
        xmin + radius + x_bundle_spacing, 
        xmax - radius,
        x_bundle_spacing)
    
    ziptie_viz_map = np.zeros((ziptie.n_bundles, ziptie.n_bundles))
    bundle_score = np.mean(
        np.matmul(x_cable[:, np.newaxis], cable_viz_map.T) *
        ziptie.cable_to_bundle_mapping)
    ziptie_viz_map[np.arange(ziptie.n_bundles), np.argsort(bundle_score)] = 1

    '''
    viz_to_i_cables = np.argsort(i_to_viz_cables)
    if ziptie.n_bundles == 0:
        x_bundles = []
        i_to_viz_bundle = []
    else:
        bundle_order = np.zeros(ziptie.n_bundles)
        
        for i_bundle, i_cables in enumerate(ziptie.bundle_to_cable_mapping):
            if len(i_cables) > 0:
                sum = 0
                for i_cable in i_cables:
                    i_cable_viz = viz_to_i_cables[i_cable]
                    sum += x_cables[i_cable_viz]
                bundle_order[i_bundle] = sum / len(i_cables)

        viz_to_i_bundle = np.argsort(bundle_order)
        i_to_viz_bundle = np.argsort(np.argsort(bundle_order))

        for i_cable_viz, activity in enumerate(ordered_activities):
            i_cable = i_to_viz_cables[i_cable_viz]
            for i_bundle in ziptie.cable_to_bundle_mapping[i_cable]:
                i_bundle_viz = viz_to_i_bundle[i_bundle]
                x_end = x_bundles[i_bundle_viz]
                x_start = x_cable[i_cable_viz]
                vt.plot_curve_activity(x_start, x_end, ymin, ymax, activity)

        for i_bundle, activity in enumerate(ziptie.bundle_activities):
            x_end = x_bundles[i_bundle_viz]
            vt.plot_point_activity(x_end, ymax, activity, x_bundle_spacing)
    '''
        for i_cable_viz, activity in enumerate(ordered_activities):
            i_cable = i_to_viz_cables[i_cable_viz]
            for i_bundle in ziptie.cable_to_bundle_mapping[i_cable]:
                i_bundle_viz = viz_to_i_bundle[i_bundle]
                x_end = x_bundles[i_bundle_viz]
                x_start = x_cable[i_cable_viz]
                vt.plot_curve_activity(x_start, x_end, ymin, ymax, activity)

        for i_bundle, activity in enumerate(ziptie.bundle_activities):
            x_end = x_bundles[i_bundle_viz]
            vt.plot_point_activity(x_end, ymax, activity, x_bundle_spacing)
    return x_bundles, bundle_viz_map
