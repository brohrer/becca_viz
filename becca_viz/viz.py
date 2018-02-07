"""
Show the world and what's going on inside the brain.
"""
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import becca.tools as bt


def plot_point_activity(x, y, activity, spacing):
    """
    Draw a point that represents the activity of a signal it carries.
    """
    r = np.minimum(np.maximum(.01, spacing / 24.), .03)
    circle = plt.Circle(
        (x, y),
        r,
        color=bt.copper_shadow,
        zorder=6.,
    )
    plt.gca().add_artist(circle)
    r_activity = activity * r * 2.
    circle = plt.Circle(
        (x, y),
        r_activity,
        color=bt.light_copper,
        zorder=7. + activity,
    )
    plt.gca().add_artist(circle)

def plot_curve_activity(x_start, y_start, x_end, y_end, activity):
    """
    Draw a smooth curve connecting two points.
    """
    t = np.arange(0., np.pi , .01)
    curve = np.cos(t)
    offset = (y_start + y_end) / 2.
    dilation = (x_end - x_start) / np.pi
    scale = (
        np.sign(x_start - x_end) *
        (y_end - y_start) / 2.
    )
    plt.plot(
        x_start + t * dilation,
        offset + curve * scale,
        color=bt.copper_shadow,
        linewidth=.8,
        zorder=2.,
    )
    plt.plot(
        x_start + t * dilation,
        offset + curve * scale,
        color=bt.light_copper,
        linewidth=activity,
        zorder=activity + 2.,
    )

def plot_line_activity(x, y, activity):
    """
    Draw a line that represents the activity of a signal it carries.
    """

    plt.plot(
        x, y,
        color=bt.copper_shadow,
        linewidth=.8,
        zorder=2.,
    )
    plt.plot(
        x, y,
        color=bt.light_copper,
        linewidth=activity,
        zorder=activity + 2.,
    )


def plot_branch(
    x_start, y_start, y_end,
    activity,
    branch_length,
    is_leaf=False,
    max_x=0.,
):
    """
    Draw the branches of the discretized sensor trees.

    @param x_start, y_start, y_end: floats
        The start and end coordinates of the branch.
    @param activity: float
        The activity level of the branch, between 0 and 1.
    @param branch_length: float
        The x distance between generations in the tree.
    @param is_leaf: boolean
        Is this branch for a leaf node?
    @param max_x: float
        In the case of a leaf node, what is the x extent of the window.
    """
    x_end = x_start + branch_length
    plot_curve_activity(x_start, y_start, x_end, y_end, activity)
    #plot_line_activity(
    #    [x_start, x_end, x_end],
    #    [y_start, y_start, y_end],
    #    activity,
    #)
    if is_leaf:
        plot_line_activity([x_end, max_x], [y_end, y_end], activity)


def brain_activity(brain=None, dpi=300):
    """
    Show what is going on in the brain at this time step.
    
    @param brain: Brain
        The brain to visualize.
    @param dpi: int
        Dots per inch for the saved image.
        300 is high resolution.
        100 is medium resolution.
        30 is low resolution.
    """
    total_width = 18.
    total_height = 9.
    outside_border = 1.
    inside_border = .75
    render_size = (total_height - 2 * outside_border - inside_border) / 2.
    model_bar = 1.
    model_size = total_height - 2 * outside_border - inside_border - model_bar
    featurizer_width = (
        total_width -
        2 * outside_border -
        2 * inside_border -
        render_size -
        model_bar -
        model_size
    )
    n_levels = 1.
    level_width = featurizer_width / (n_levels + 1)

    frame_linewidth = .3

    frame_color = bt.copper
    highlight = .05

    fig = plt.figure(num=84782, figsize=(18., 9.))
    fig.clf()
    ax = plt.gca()
    ax.add_patch(patches.Rectangle(
        (0., 0.),
        total_width,
        total_height,
        facecolor=bt.copper_shadow,
        edgecolor='none',
        zorder=-16.,
    ))

    def box(xmin, xmax, ymin, ymax, color='purple', linewidth=1.):
        plt.plot(
            [xmin, xmin, xmax, xmax, xmin],
            [ymin, ymax, ymax, ymin, ymin],
            color=color,
            linewidth=linewidth,
            zorder=5.
        )
        plt.imshow(
            np.zeros((100, 100)),
            cmap=plt.get_cmap('inferno'),
            zorder=-10.,
            extent=[xmin, xmax, ymin, ymax],
        )

    # Sensor render
    xmin = outside_border
    xmax = xmin + render_size
    ymin = outside_border
    ymax = ymin + render_size
    box(
        xmin, xmax, ymin, ymax,
        color=frame_color,
        linewidth=frame_linewidth,
    )

    # World render
    ymin = ymax + inside_border
    ymax = ymin + render_size
    box(
        xmin, xmax, ymin, ymax,
        color=frame_color,
        linewidth=frame_linewidth,
    )

    # Discretizer
    xmin = xmax + inside_border
    xmax = xmin + level_width
    ymin = outside_border
    ymax = total_height - outside_border
    # Make the frame.
    box(
        xmin, xmax, ymin, ymax,
        color=frame_color,
        linewidth=frame_linewidth,
    )
    # Gather all the nodes, so they can be sorted by position.
    # Start by setting aside positions for the actions at the top
    i_actions = np.cumsum(np.ones(brain.n_actions))
    positions = list(i_actions - 1e6)
    # then collect positions from all the discretized sensors.
    for discretizer in brain.discretizers:
        positions += [node.position for node in
                      discretizer.numeric_cats.get_list()]
        positions.append(discretizer.position)
        positions += [node.position for node in
                      discretizer.string_cats.get_list()]
    positions = np.array(positions)
    sorted_positions = np.sort(positions)
    y_spacing = np.minimum((ymax - ymin) / (float(len(positions)) + 1.),
                           level_width)
    node_y = ymax - y_spacing * np.cumsum(np.ones(len(positions)))

    def get_y(node):
        """
        Find the y position that should be associated with a node.

        @param node: an object with a position member
            Both Nodes and Discretizers fit this description.

        @return y_position: float
        """
        i_position = np.where(sorted_positions == node.position)
        return node_y[i_position]

    y_inputs = np.zeros(brain.max_n_inputs)
    for i_action, activity in enumerate(brain.previous_actions):
        y_inputs[i_action] = node_y[i_action]
        plot_line_activity(
            [xmin, xmax],
            [node_y[i_action], node_y[i_action]],
            activity,
        )

    for discretizer in brain.discretizers:
        # Build trees.
        n_depth = 1. + np.maximum(
            discretizer.numeric_cats.depth,
            discretizer.string_cats.depth)
        branch_length = level_width / n_depth
        sensor_y = get_y(discretizer)

        def plot_tree(tree):
            root_y = get_y(tree.root)
            y_inputs[tree.root.i_input] = root_y
            activity = brain.input_activities[tree.root.i_input]
            plot_branch(xmin, sensor_y, root_y, activity, branch_length)
            nodes = tree.get_list()
            for node in nodes:
                if node.lo_child is not None:
                    x_start = xmin + (node.depth + 1.) * branch_length
                    y_start = get_y(node)
                    y_end = get_y(node.lo_child)
                    y_inputs[node.lo_child.i_input] = y_end
                    lo_activity = brain.input_activities[node.lo_child.i_input]
                    plot_branch(
                        x_start,
                        y_start,
                        y_end,
                        lo_activity,
                        branch_length,
                        node.lo_child.leaf,
                        xmax,
                    )
                    plot_point_activity(
                        x_start + branch_length,
                        y_end,
                        lo_activity,
                        y_spacing,
                    )
                    y_end = get_y(node.hi_child)
                    y_inputs[node.hi_child.i_input] = y_end
                    hi_activity = brain.input_activities[node.hi_child.i_input]
                    plot_branch(
                        x_start,
                        y_start,
                        y_end,
                        hi_activity,
                        branch_length,
                        node.hi_child.leaf,
                        xmax,
                    )
                    plot_point_activity(
                        x_start + branch_length,
                        y_end,
                        hi_activity,
                        y_spacing,
                    )

        # Build string branch.
        plot_tree(discretizer.string_cats)
        plot_tree(discretizer.numeric_cats)

    model_order = np.cumsum(np.ones(
        brain.model.feature_activities.size, dtype=np.int)) - 1
    # Start at index 2 to account for the fact that two additional actions are
    # added internally in the model: a 'do nothing' and a 'do everything'.

    i_actions = np.cumsum(np.ones(brain.n_actions)) - 1
    input_positions = list(i_actions - 1e6)
    i_inputs = list(i_actions)
    for discretizer in brain.discretizers:
        input_positions += [node.position for node in
                            discretizer.numeric_cats.get_list()]
        input_positions += [node.position for node in
                            discretizer.string_cats.get_list()]
        i_inputs += [node.i_input for node in
                     discretizer.numeric_cats.get_list()]
        i_inputs += [node.i_input for node in
                     discretizer.string_cats.get_list()]
    # The nodes are retrieved in arbitrary order.
    # i_inputs their order in the input array.
    input_positions = np.array(input_positions)
    i_inputs = np.array(i_inputs).astype(np.int)
    retrieved_positions = input_positions[np.argsort(i_inputs)]
    input_order = np.argsort(retrieved_positions)

    i_last = 2
    i_start = i_last
    i_last = i_start + input_order.size
    model_order[i_start:i_last] = input_order + i_start
    i_last = 2 + brain.max_n_inputs


    # Levels
    y_cables = y_inputs
    cable_spacing = y_spacing
    # for i_level in range(n_levels):
    for _ in range(1):
        # Create output re-mapping for visual cleanliness.

        xmin = xmax
        xmax = xmin + level_width
        box(
            xmin, xmax, ymin, ymax,
            color=frame_color,
            linewidth=frame_linewidth,
        )
        ziptie = brain.featurizer.ziptie
        n_cables = ziptie.max_n_cables
        n_bundles = ziptie.n_bundles
        # Create circles for cables.
        for i_cable in range(n_cables):
            cable_activity = ziptie.cable_activities[i_cable]
            y_cable = y_cables[i_cable]
            plot_point_activity(xmin, y_cable, cable_activity, cable_spacing)

        bundle_spacing = np.minimum((ymax - ymin) / (n_bundles + 1.),
                                    level_width)

        # Organize the bundles to be near their constituent cables.
        bundle_positions = np.zeros(n_bundles)
        cables_per_bundle = np.zeros(n_bundles)
        summed_cable_positions = np.zeros(n_bundles)
        for i_conn in range(ziptie.n_map_entries):
            i_cable = ziptie.bundle_map_cols[i_conn]
            i_bundle = ziptie.bundle_map_rows[i_conn]
            y_cable = y_cables[i_cable]
            cables_per_bundle[i_bundle] += 1.
            summed_cable_positions[i_bundle] += y_cable
        bundle_positions = summed_cable_positions / (cables_per_bundle + 1e-6)
        # Do magic with indices.
        # If this is opaque, it it because I found it confusing.
        # The code here is the result of two days' trial and error.
        # I really need to wrap my head around sorting and indexing.
        i_position_sort = np.argsort(bundle_positions)
        i_bundle_order = np.argsort(i_position_sort)
        y_bundles_unsorted = (
            ymin +  np.cumsum(np.ones(n_bundles)) * bundle_spacing)
        y_bundles = y_bundles_unsorted[i_bundle_order]
        # These indices bound the bundle activities for this level.
        i_start = i_last
        i_last = i_start + n_bundles
        model_order[i_start:i_last] = i_position_sort[::-1] + i_start

        # Connect cables with their bundles.
        for i_conn in range(ziptie.n_map_entries):
            i_cable = ziptie.bundle_map_cols[i_conn]
            i_bundle = ziptie.bundle_map_rows[i_conn]
            y_cable = y_cables[i_cable]
            y_bundle = y_bundles[i_bundle]
            activity = (ziptie.bundle_activities[i_bundle] *
                        ziptie.cable_activities[i_cable])
            plot_curve_activity(xmin, y_cable, xmax, y_bundle, activity)

        y_cables = y_bundles
        cable_spacing = bundle_spacing

    # Create circles for the final set of bundles.
    for i_bundle in range(n_bundles):
        bundle_activity = ziptie.bundle_activities[i_bundle]
        y_bundle = y_bundles[i_bundle]
        plot_point_activity(xmax, y_bundle, bundle_activity, bundle_spacing)

    # Model
    xmax = total_width - outside_border
    xmin = xmax - model_size
    ymin = outside_border
    ymax = ymin + model_size
    # Rewards are green.
    # Negative rewards (punishments) are red.
    # Curiosity is blue.
    reward_color = [0., 158. / 255., 115. / 255.]
    punishment_color = [230. / 255., 159. / 255., 0.]
    curiosity_color = [86. / 255., 180. / 255., 233. / 255.]
    reward_image = np.concatenate([
        reward_color[0] * brain.model.prefix_rewards[:, :, np.newaxis],
        reward_color[1] * brain.model.prefix_rewards[:, :, np.newaxis],
        reward_color[2] * brain.model.prefix_rewards[:, :, np.newaxis],
        ], axis=2)
    punishment_image = np.concatenate([
        -punishment_color[0] * brain.model.prefix_rewards[:, :, np.newaxis],
        -punishment_color[1] * brain.model.prefix_rewards[:, :, np.newaxis],
        -punishment_color[2] * brain.model.prefix_rewards[:, :, np.newaxis],
        ], axis=2)
    curiosity_image = np.concatenate([
        curiosity_color[0] * brain.model.prefix_curiosities[:, :, np.newaxis],
        curiosity_color[1] * brain.model.prefix_curiosities[:, :, np.newaxis],
        curiosity_color[2] * brain.model.prefix_curiosities[:, :, np.newaxis],
        ], axis=2)
    model_image = (
        np.maximum(reward_image, 0.) +
        np.maximum(punishment_image, 0.) +
        np.maximum(curiosity_image, 0.)
    )
    

    # Rearrange the inputs in the model image to match the levels.
    model_image = model_image[model_order, :, :]
    model_image = model_image[:, model_order, :]
    # Highlight actions and derived features.
    model_image[2:brain.n_actions + 2, :, :] += highlight
    model_image[:, 2:brain.n_actions + 2, :] += highlight
    model_image[brain.max_n_inputs + 2:, :, :] += highlight
    model_image[:, brain.max_n_inputs + 2:, :] += highlight
    model_image = np.minimum(1., model_image)
    plt.imshow(
        model_image,
        extent=[xmin, xmax, ymin, ymax],
        interpolation='nearest',
        origin='upper',
        zorder=3.,
        vmin=0.,
        vmax=1.,
    )
    box(
        xmin, xmax, ymin, ymax,
        color=frame_color,
        linewidth=frame_linewidth,
    )

    # Model features
    xmax = xmin
    xmin = xmax - model_bar
    features_image = np.zeros((brain.model.n_features, 100))
    features = brain.model.feature_activities
    for i_feature, feature in enumerate(features):
        feature = np.minimum(.999, np.maximum(0., feature))
        features_image[i_feature, -int(feature * 100.):] = feature
    features_image = features_image[model_order, :]
    features_image[:brain.n_actions + 2, :] += highlight
    features_image[brain.max_n_inputs + 2:, :] += highlight
    plt.imshow(
        features_image,
        cmap=plt.get_cmap('inferno'),
        interpolation='nearest',
        extent=[xmin, xmax, ymin, ymax],
        origin='upper',
        vmax=1.,
        vmin=0.,
        zorder=3.,
    )
    box(
        xmin, xmax, ymin, ymax,
        color=frame_color,
        linewidth=frame_linewidth,
    )
    # Model goals
    xmin = xmax
    xmax = xmax + model_size
    ymin = ymax
    ymax = ymin + model_bar
    goals_image = np.zeros((100, brain.model.n_features))
    goals = brain.model.feature_goal_votes
    for i_goal, goal in enumerate(goals):
        goal = np.minimum(.999, np.maximum(0., goal))
        goals_image[:int(goal * 100.), i_goal] = goal
    goals_image = goals_image[:, model_order]
    goals_image[:, 2:brain.n_actions + 2] += highlight
    goals_image[:, brain.max_n_inputs + 2:] += highlight
    plt.imshow(
        goals_image,
        cmap=plt.get_cmap('inferno'),
        interpolation='nearest',
        extent=[xmin, xmax, ymin, ymax],
        origin='lower',
        vmax=1.,
        vmin=0.,
        zorder=3.,
    )
    box(
        xmin, xmax, ymin, ymax,
        color=frame_color,
        linewidth=frame_linewidth,
    )

    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')

    filename = 'becca_{0}.png'.format(brain.name)
    pathname = os.path.join(brain.log_dir, filename)
    plt.savefig(pathname, format='png', dpi=dpi)


def viz_sensing(brain):
    """
    Turn the most recent
    """


if __name__ == '__main__':
    # brain_activity()
    sense()
