"""
Show what's going on inside an input filter.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import becca_viz.viz_tools as vt


def render(filter, bbox, x_pool_prev, pool_viz_map, y_prev,  radius=0):
    """
    Make a picture of the discretization that happens in an input filter.

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
    frame_height = ymax - ymin

    n_pool = x_pool_prev.size
    x_pool_spacing = (frame_width - 2 * radius) / (x_pool_prev.size + 1)
    x_pool = np.arange(
        xmin + radius + x_pool_spacing, 
        xmax - radius,
        x_pool_spacing)

    pool_viz_activities = np.matmul(filter.candidate_activities, pool_viz_map)

    # Connect the previous block(s) to this one.
    for i_pool_viz, activity in enumerate(pool_viz_activities): 
        vt.plot_curve_activity(
            x_pool_prev[i_pool_viz], x_pool[i_pool_viz],
            y_prev, ymin, activity)
        vt.plot_point_activity(
            x_pool[i_pool_viz], ymin, activity, x_pool_spacing)
    
    pool_to_input_map = filter.mapping[:n_pool, :filter.n_inputs]
    input_to_pool_viz_map = np.matmul(
        pool_to_input_map.T, pool_viz_map)
    i_pool_viz_in_use = np.where(input_to_pool_viz_map)[0]
    pool_to_input_viz_map = (
        np.eye(n_pool, dtype=np.int)[:, i_pool_viz_in_use])
    n_in_use = i_pool_viz_in_use.size

    x_spacing = (frame_width - 2 * radius) / (n_in_use + 1)
    x_inputs = np.arange(
        xmin + radius + x_spacing, 
        xmax - radius,
        x_spacing)

    input_viz_map = np.matmul(input_to_pool_viz_map, pool_to_input_viz_map)
    input_activities = np.matmul(
        filter.candidate_activities, pool_to_input_map)
    input_viz_activities = np.matmul(
        pool_viz_activities, pool_to_input_viz_map)

    # Show the filter's selection
    for i_input_viz, activity in enumerate(input_viz_activities):
        i_pool_viz = np.where(pool_to_input_viz_map[:, i_input_viz])[0]
        x_end = x_inputs[i_input_viz]
        x_start = x_pool[i_pool_viz]
        vt.plot_curve_activity(x_start, x_end, ymin, ymax, activity)
        vt.plot_point_activity(x_end, ymax, activity, x_spacing)
        
    return x_inputs, input_viz_map
