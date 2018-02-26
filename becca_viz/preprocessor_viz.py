"""
Show what's going on inside the preprocessor.
"""
import os

import matplotlib.pyplot as plt
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
    x_inputs: array of floats
        The absolute x positions of the input nodes.
    i_to_viz: array of ints
        The indices for ordering input activities to align with
        the visualization.
    """

    xmin, xmax, ymin, ymax = bbox
    frame_width = xmax - xmin
    frame_height = ymax - ymin
    # Collect positions from all the discretized sensors.
    positions = []
    i_pool = []
    for discretizer in preprocessor.discretizers:
        num_tree_list = discretizer.numeric_cats.get_list()
        str_tree_list = discretizer.string_cats.get_list()
        positions += [node.position for node in num_tree_list]
        positions.append(discretizer.position)
        positions += [node.position for node in str_tree_list]
        i_pool += [node.i_input for node in num_tree_list]
        i_pool.append(-1)
        i_pool += [node.i_input for node in str_tree_list]
    positions = np.array(positions)
    i_pool = np.array(i_pool)

    n_pool = i_pool.size

    # pool_order = np.argsort(i_pool)
    # pool_by_pool = i_pool[pool_order]
    # positions_by_pool = positions[pool_order]
    # pool_to_viz_order = np.argsort(positions_by_pool)
    # pool_to_viz[(np.arange(n_pool), pool_to_viz_order)] = 1
    position_order = np.argsort(positions)
    positions_by_positions = positions[position_order]
    pool_by_positions = i_pool[position_order]
    i_active_positions = np.where(pool_by_positions >= 0)[0]
    active_pool_by_positions = pool_by_positions[i_active_positions]
    active_positions_by_positions = positions_by_positions[i_active_positions]
    pool_order = np.argsort(active_pool_by_positions)
    position_order = np.argsort(np.argsort(active_positions_by_positions))
    pool_to_viz = np.zeros((i_active_positions.size, i_active_positions.size))
    pool_to_viz[(pool_order, position_order)] = 1

    x_spacing = (frame_width - 2 * radius) / float(len(positions) + 1)
    node_x = xmin + radius + x_spacing * np.cumsum(np.ones(len(positions)))
    active_node_x = node_x[i_active_positions]

    def get_x(node):
        """
        Find the x position that should be associated with a node.
        
        Parameters
        ----------
        node: an object with a position member
            Both Nodes and Discretizers fit this description.

        Returns
        -------
        x_position: float
        """
        i_position = np.where(positions_by_positions == node.position)
        return node_x[i_position]

    x_inputs = np.zeros(preprocessor.n_inputs)
    
    for discretizer in preprocessor.discretizers:
        # Build trees.
        n_depth = 1. + np.maximum(
            discretizer.numeric_cats.depth,
            discretizer.string_cats.depth)
        branch_length = frame_height / n_depth
        sensor_x = get_x(discretizer)

        def plot_tree(tree):
            root_x = get_x(tree.root)
            x_inputs[tree.root.i_input] = root_x
            activity = preprocessor.input_activities[tree.root.i_input]
            plot_branch(sensor_x, root_x, ymin, activity, branch_length)
            nodes = tree.get_list()
            for node in nodes:
                if node.lo_child is not None:
                    y_start = ymin + (node.depth + 1.) * branch_length
                    x_start = get_x(node)
                    x_end = get_x(node.lo_child)
                    x_inputs[node.lo_child.i_input] = x_end
                    lo_activity = preprocessor.input_activities[
                        node.lo_child.i_input]
                    plot_branch(
                        x_start,
                        x_end,
                        y_start,
                        lo_activity,
                        branch_length,
                        node.lo_child.leaf,
                        ymax,
                    )
                    vt.plot_point_activity(
                        x_end,
                        y_start + branch_length,
                        lo_activity,
                        x_spacing,
                    )
                    x_end = get_x(node.hi_child)
                    x_inputs[node.hi_child.i_input] = x_end
                    hi_activity = preprocessor.input_activities[
                        node.hi_child.i_input]
                    plot_branch(
                        x_start,
                        x_end,
                        y_start,
                        hi_activity,
                        branch_length,
                        node.hi_child.leaf,
                        ymax,
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

    return active_node_x, pool_to_viz

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

