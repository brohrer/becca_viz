"""
Show what's going on inside the preprocessor.
"""
import numpy as np

import becca_viz.viz_tools as vt


def render(preprocessor, bbox, radius=0):
    """
    Make a picture of the discretization that happens in the Preprocessor.

    Parameters
    ----------
    bbox: list of floats
        Of the form [x_bottom, x_top, y_left, y_right]
    preprocessor: Preprocessor

    Returns
    -------
    input_x_viz: array of floats
        The absolute x positions of the input nodes.
    input_viz_map: 2D array of ints
        A map from the order the inputs occur to the visualization order.
    """
    xmin, xmax, ymin, ymax = bbox
    frame_width = xmax - xmin
    frame_height = ymax - ymin

    n_inputs = preprocessor.n_inputs
    n_disc = len(preprocessor.discretizers)
    # The number of inputs plus the number of discretizers.
    n_input_d = n_inputs + n_disc
    positions = np.zeros(n_input_d)

    x_spacing = (frame_width - 2 * radius) / float(len(positions) + 1)
    node_x = xmin + radius + x_spacing * np.cumsum(np.ones(len(positions)))

    # Get the x positions of all the nodes.
    # This needs to be in the same order as activities.
    i_disc = 0
    for i_disc, discretizer in enumerate(preprocessor.discretizers):
        positions[i_disc] = discretizer.position
        num_tree_list = discretizer.numeric_cats.get_list()
        str_tree_list = discretizer.string_cats.get_list()
        for node in num_tree_list:
            positions[node.i_input + n_disc] = node.position
        for node in str_tree_list:
            positions[node.i_input + n_disc] = node.position

    position_order = np.argsort(positions)
    i_input_d_by_position = np.arange(n_input_d)[position_order]
    x_by_i_input_d = node_x[np.argsort(i_input_d_by_position)]
    discretizer_x = x_by_i_input_d[:n_disc]
    input_x = x_by_i_input_d[n_disc:]
    input_viz_map = np.zeros((n_inputs, n_inputs), dtype=np.int)
    input_viz_map[np.argsort(input_x), np.arange(n_inputs)] = 1
    input_x_viz = np.matmul(input_x, input_viz_map)

    for i_discretizer, discretizer in enumerate(preprocessor.discretizers):
        # Build trees.
        n_depth = 1. + np.maximum(
            discretizer.numeric_cats.depth,
            discretizer.string_cats.depth)
        branch_length = frame_height / n_depth
        sensor_x = discretizer_x[i_discretizer]

        def plot_tree(tree):
            root_x = input_x[tree.root.i_input]
            activity = preprocessor.input_activities[tree.root.i_input]
            plot_branch(sensor_x, root_x, ymin, activity, branch_length)
            nodes = tree.get_list()
            for node in nodes:
                if node.lo_child is not None:
                    y_start = ymin + (node.depth + 1.) * branch_length
                    x_start = input_x[node.i_input]
                    x_end = input_x[node.lo_child.i_input]
                    lo_activity = preprocessor.input_activities[
                        node.lo_child.i_input]
                    plot_branch(
                        x_start,
                        x_end,
                        y_start,
                        lo_activity,
                        branch_length,
                        is_leaf=node.lo_child.leaf,
                        max_y=ymax,
                    )
                    vt.plot_point_activity(
                        x_end,
                        y_start + branch_length,
                        lo_activity,
                        x_spacing,
                    )
                    x_end = input_x[node.hi_child.i_input]
                    hi_activity = preprocessor.input_activities[
                        node.hi_child.i_input]
                    plot_branch(
                        x_start,
                        x_end,
                        y_start,
                        hi_activity,
                        branch_length,
                        is_leaf=node.hi_child.leaf,
                        max_y=ymax,
                    )
                    vt.plot_point_activity(
                        x_end,
                        y_start + branch_length,
                        hi_activity,
                        x_spacing,
                    )

        # Build string branch.
        plot_tree(discretizer.string_cats)
        plot_tree(discretizer.numeric_cats)

    return input_x_viz, input_viz_map


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
