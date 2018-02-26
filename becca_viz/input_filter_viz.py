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

    x_pool_spacing = (frame_width - 2 * radius) / (x_pool_prev.size + 1)
    x_pool = np.arange(
        xmin + radius + x_pool_spacing, 
        xmax - radius,
        x_pool_spacing)

    # Connect the previous block(s) to this one.
    for i_candidate, activity in enumerate(filter.candidate_activities): 
        vt.plot_curve_activity(
            x_pool_prev[i_candidate], x_pool[i_candidate],
            y_prev, ymin, activity)
        vt.plot_point_activity(
            x_pool[i_candidate], ymin, activity, x_pool_spacing)

    i_pool_in_use = np.where(filter.mapping)[0]
    n_in_use = i_pool_in_use.size

    x_spacing = (frame_width - 2 * radius) / (n_in_use + 1)
    x_inputs = np.arange(
        xmin + radius + x_spacing, 
        xmax - radius,
        x_spacing)

    input_viz_map = np.matmul(filter.mapping[:filter.n_candidates, :].T,
                              pool_viz_map)[:,i_pool_in_use]
    # Show the filter's selection
    i_to_viz_pool = np.where(pool_viz_map)[1]
    i_to_viz_input = np.where(input_viz_map)[1]

    for i_pool in i_pool_in_use:
        activity = filter.candidate_activities[i_pool]
        i_pool_viz = i_to_viz_pool[i_pool]
        i_input = np.where(filter.mapping[i_pool])[0]
        i_input_viz = i_to_viz_input[i_input]
        x_end = x_inputs[i_input_viz]
        x_start = x_pool[i_pool_viz]
        vt.plot_curve_activity(x_start, x_end, ymin, ymax, activity)
        vt.plot_point_activity(x_end, ymax, activity, x_spacing)
    '''
    
    for i_pool, activity in enumerate(filter.candidate_activities):
        i_input = filter.mapping[i_pool]
        i_pool_viz = pool_viz_map[i_pool]
        if i_input >= 0:
            i_input_viz = i_to_viz_input[i_input]
            x_end = x_inputs[i_input_viz]
            x_start = x_pool[i_pool_viz]
            vt.plot_curve_activity(x_start, x_end, ymin, ymax, activity)
            vt.plot_point_activity(x_end, ymax, activity, x_spacing)
    '''
        
    return x_inputs, input_viz_map
