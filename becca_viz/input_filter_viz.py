"""
Show what's going on inside an input filter.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import becca_viz.viz_tools as vt


def render(filter, bbox, x_pool_prev, i_to_viz_pool, y_prev,  radius=0):
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
    i_to_viz_cables: array of ints
        The indices for ordering cables to align with
        the visualization.
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
    ordered_activities = filter.candidate_activities[i_to_viz_pool]
    
    for i_act, activity in enumerate(ordered_activities):
        vt.plot_curve_activity(
            x_pool_prev[i_act], x_pool[i_act], y_prev, ymin, activity)
        vt.plot_point_activity(
            x_pool[i_act], ymin, activity, x_pool_spacing)

    i_pool_in_use = np.where(filter.mapping >= 0)[0]
    n_in_use = i_pool_in_use.size
    in_use_mask = np.zeros(x_pool.size)
    in_use_mask[i_pool_in_use] = 1

    x_spacing = (frame_width - 2 * radius) / (n_in_use + 1)
    x_inputs = np.arange(
        xmin + radius + x_spacing, 
        xmax - radius,
        x_spacing)

    in_use_activities = filter.candidate_activities[i_pool_in_use]
    mapping = filter.mapping[i_pool_in_use]
    input_activities = in_use_activities[mapping]

    # inverse_mapping = filter.inverse_mapping
    i_to_viz_input = -np.ones(filter.n_inputs)
    i_to_viz_input_order = -np.ones(filter.n_inputs)

    # Show the filter's selection
    # viz_to_i_pool = np.argsort(i_to_viz_pool)
    # viz_to_i_input = np.argsort(i_to_viz_input)
    # for i_pool, activity in enumerate(filter.candidate_activities):
    for i_pool_viz, activity in enumerate(ordered_activities):
        i_pool = i_to_viz_pool[i_pool_viz]
        i_input = filter.mapping[i_pool]
        if i_input >= 0:
            i_to_viz_input_order[i_input] = i_pool_viz

    i_to_viz_input_order = i_to_viz_input_order[
        np.where(i_to_viz_input_order >= 0)]
    i_to_viz_input = np.argsort(np.argsort(i_to_viz_input_order))

    for i_pool_viz, activity in enumerate(ordered_activities):
        i_pool = i_to_viz_pool[i_pool_viz]
        i_input = filter.mapping[i_pool]
        if i_input >= 0:
            i_input_viz = i_to_viz_input[i_input]
            x_end = x_inputs[i_input_viz]
            x_start = x_pool[i_pool_viz]
            vt.plot_curve_activity(x_start, x_end, ymin, ymax, activity)
            vt.plot_point_activity(x_end, ymax, activity, x_spacing)
    
    return x_inputs, i_to_viz_input


def plot_branch(
    x_start, x_end,
    y_start,
    activity,
    branch_length,
    is_leaf=False,
    max_y=0.,
):
    """
    Draw the branches of the discretized sensor trees.

    @param y_start, x_start, x_end: floats
        The start and end coordinates of the branch.
    @param activity: float
        The activity level of the branch, between 0 and 1.
    @param branch_length: float
        The y distance between generations in the tree.
    @param is_leaf: boolean
        Is this branch for a leaf node?
    @param max_y: float
        In the case of a leaf node, what is the x extent of the window.
    """
    y_end = y_start + branch_length
    vt.plot_curve_activity(x_start, x_end,  y_start, y_end, activity)
    if is_leaf:
        vt.plot_line_activity([x_end, x_end], [y_end, max_y], activity)

