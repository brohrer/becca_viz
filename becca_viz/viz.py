"""
Show the world and what's going on inside the brain.
"""
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import becca_viz.viz_tools as vt
import becca_viz.preprocessor_viz as preprocessor_viz
import becca_viz.postprocessor_viz as postprocessor_viz
import becca_viz.input_filter_viz as input_filter_viz
# import becca_viz.ziptie_viz as ziptie_viz


lwf = .3  # Linewidth of the frames

wd_ = 18  # Total width of the image
ht_ = 9  # Total heigh of the image
brd = .75  # Border thickness around frames

halfwidth_weight = 5
centerwidth_weight = 1
total_weight = 2 * halfwidth_weight + centerwidth_weight

no_borders = wd_ - brd * 4  # Width available after borders are removed
wdh = no_borders * halfwidth_weight / total_weight  # Width of wide columns
wdc = no_borders * centerwidth_weight / total_weight  # Width of center column

htf = ht_ - brd * 2  # Full height, minus borders
n_zipties = 2  # Assumed number of zipties
n_rows = 1 + n_zipties * 2  # Total number of frames stacked vertically
no_borders_ht = ht_ - brd * (n_rows + 1)  # Total non-border height
htr = no_borders_ht / n_rows  # height of each row
wdq = (wdh - brd) / 2  # Quarter-width

rad = htr / 6  # The radius of the rounded corners on frames

preprocessor_bbox = [brd, brd + wdq,
                     brd, brd + htr]
postprocessor_bbox = [2 * brd + wdq, brd + wdh,
                      brd, brd + htr]
filt_0_bbox = [brd, brd + wdh,
                     2 * brd + htr, 2 * brd + 2 * htr]
ziptie_0_bbox = [brd, brd + wdh,
                 3 * brd + 2 * htr, 3 * brd + 3 * htr]
cable_filt_1_bbox = [brd, brd + wdh,
                     4 * brd + 3 * htr, 4 * brd + 4 * htr]
ziptie_1_bbox = [brd, brd + wdh,
                 5 * brd + 4 * htr, 5 * brd + 5 * htr]
feature_filt_bbox = [2 * brd + wdh, 2 * brd + wdh + wdc,
                     brd, brd + htf]
model_bbox = [3 * brd + wdh + wdc, 3 * brd + 2 * wdh + wdc,
              brd, brd + htf]


def visualize(brain):
    """
    Render the sensor information making its way up through the brain.

    Parameters
    ----------
    brain: Brain
    """
    create_background()
    x_inputs, preprocessor_viz_map = preprocessor_viz.render(
        brain.preprocessor, preprocessor_bbox, radius=rad)
    x_commands, postprocessor_viz_map = postprocessor_viz.render(
        brain.postprocessor, postprocessor_bbox, radius=rad)
    n_pre = preprocessor_viz_map.shape[0]
    n_post = postprocessor_viz_map.shape[0]

    pool_0_viz_map = np.block([
        [np.zeros((n_post, n_pre)), postprocessor_viz_map],
        [preprocessor_viz_map, np.zeros((n_pre, n_post))]])
    # TODO: handle multiple zipties
    x_pool_0 = np.concatenate((x_inputs, x_commands))
    x_cables_0, cables_0_viz_map = input_filter_viz.render(
        brain.featurizer.filter,
        filt_0_bbox,
        x_pool_0,
        pool_0_viz_map,
        preprocessor_bbox[3],  # max y value of the Preprocessor
        radius=rad)
    # x_pool_1, pool_1_viz_map = ziptie_viz.render(
    #     brain.featurizer.ziptie,
    #     ziptie_0_bbox,
    #     x_cables_0,
    #     cables_0_viz_map,
    #     filt_0_bbox[3],  # max y value of the InputFilter
    #     radius=rad)

    finalize(brain, dpi=300)


def create_background():
    """
    Set up the backdrop for the visualization.
    """
    fig = plt.figure(num=84782, figsize=(wd_, ht_))
    fig.clf()
    ax = plt.gca()
    ax.add_patch(patches.Rectangle(
        (0, 0),
        wd_,
        ht_,
        facecolor=vt.dark_grey,
        edgecolor='none',
        zorder=-16.,
    ))
    # Preprocessor frame
    vt.draw_frame(
        bbox=preprocessor_bbox,
        radius=rad,
        facecolor=vt.dark_grey,
        edgecolor=vt.copper,
    )            
    # Postprocessor frame
    vt.draw_frame(
        bbox=postprocessor_bbox,
        radius=rad,
        facecolor=vt.dark_grey,
        edgecolor=vt.copper,
    )            
    # Cable filter 0 frame
    vt.draw_frame(
        bbox=filt_0_bbox,
        radius=rad,
        facecolor=vt.dark_grey,
        edgecolor=vt.copper,
    )            
    # Ziptie 0 frame
    vt.draw_frame(
        bbox=ziptie_0_bbox,
        radius=rad,
        facecolor=vt.dark_grey,
        edgecolor=vt.copper,
    )            
    # Cable filter 1 frame
    vt.draw_frame(
        bbox=cable_filt_1_bbox,
        radius=rad,
        facecolor=vt.dark_grey,
        edgecolor=vt.copper,
    )            
    # Ziptie 1 frame
    vt.draw_frame(
        bbox=ziptie_1_bbox,
        radius=rad,
        facecolor=vt.dark_grey,
        edgecolor=vt.copper,
    )            
    # feature filter frame
    vt.draw_frame(
        bbox=feature_filt_bbox,
        radius=rad,
        facecolor=vt.dark_grey,
        edgecolor=vt.copper,
    )            
    # model frame
    vt.draw_frame(
        bbox=model_bbox,
        radius=rad,
        facecolor=vt.dark_grey,
        edgecolor=vt.copper,
    )            

    return


def finalize(brain, dpi=300, phase=1):
    """
    Complete any final formatting and save a copy of the figure.

    Parameters
    ----------
    braind: Brain
    dpi: int
        The dots per inch for the saved figure.
    phase: int
        During each time step the brain does a lot and it's tough to get
        it all into one image. To handle this, it is broken up into phases.
        Phase 1 is sensing (an upward pass through the architecture) and
        Phase 2 is acting ( a downward pass through the architecture).

    """
    plt.tight_layout()
    # plt.axis('equal')
    plt.axis('off')

    filename = 'becca_{name}_{dpi:04d}_{age:08d}_{phase}.png'.format(
        name=brain.name, age=brain.timestep, dpi=dpi, phase=phase)
    pathname = os.path.join(brain.log_dir, filename)
    plt.savefig(pathname, format='png', dpi=dpi)


'''
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

    frame_color = vt.copper
    highlight = .05

    fig = plt.figure(num=84782, figsize=(18., 9.))
    fig.clf()
    ax = plt.gca()
    ax.add_patch(patches.Rectangle(
        (0., 0.),
        total_width,
        total_height,
        facecolor=vt.copper_shadow,
        edgecolor='none',
        zorder=-16.,
    ))

    def box(xmin, xmax, ymin, ymax, color=, linewidth=1.):
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
'''
