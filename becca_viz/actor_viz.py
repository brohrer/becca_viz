import numpy as np

import becca_viz.viz_tools as vt

n_image_rows = 5


def render(model, actor, bbox, viz_map, max_floor=1e-2, radius=0):
    """
    Turn it into a picture.
    """
    xmin, xmax, ymin, ymax = bbox
    frame_width = xmax - xmin
    frame_height = ymax - ymin

    y_gap = 0
    im_height = (frame_height - (n_image_rows + 1) * y_gap) / n_image_rows
    x_gap = (frame_width - im_height) / 2

    conditional_rewards_viz = vt.nd_map(
        model.conditional_rewards[2:], viz_map)
    title = ("Conditional rewards")
    vt.scatter_1D(
        conditional_rewards_viz,
        x0=xmin + x_gap,
        y0=ymin + 5 * y_gap + 4.5 * im_height,
        width=im_height,
        xlabel='goals',
        title=title,
        autoscale=True,
    )

    conditional_curiosities_viz = vt.nd_map(
        model.conditional_curiosities[2:], viz_map)
    title = ("Conditional curiosities")
    vt.scatter_1D(
        conditional_curiosities_viz,
        x0=xmin + x_gap,
        y0=ymin + 4 * y_gap + 3.5 * im_height,
        width=im_height,
        xlabel='goals',
        title=title,
        autoscale=True,
    )

    conditional_goal_rewards_raw = np.max(
        model.conditional_predictions *
        model.goal_activities[np.newaxis, :], axis=1)
    conditional_goal_rewards_viz = vt.nd_map(
        conditional_goal_rewards_raw[2:], viz_map)
    title = ("Conditional goal_rewards")
    vt.scatter_1D(
        conditional_goal_rewards_viz,
        x0=xmin + x_gap,
        y0=ymin + 3 * y_gap + 2.5 * im_height,
        width=im_height,
        xlabel='goals',
        title=title,
        autoscale=True,
    )

    goal_activities_viz = vt.nd_map(
        actor.previous_goal_collection[2:], viz_map)
    title = ("Goal_collection before")
    vt.scatter_1D(
        goal_activities_viz,
        x0=xmin + x_gap,
        y0=ymin + 2 * y_gap + 1.5 * im_height,
        width=im_height,
        xlabel='goals',
        title=title,
        autoscale=True,
    )

    goal_activities_viz = vt.nd_map(actor.goal_collection[2:], viz_map)
    title = ("Goal_collection after")
    vt.scatter_1D(
        goal_activities_viz,
        x0=xmin + x_gap,
        y0=ymin + 1 * y_gap + 0.5 * im_height,
        width=im_height,
        xlabel='goals',
        title=title,
        autoscale=True,
    )
    return


def render_structure(model, bbox, viz_map, radius=0):
    """

    """
    xmin, xmax, ymin, ymax = bbox
    frame_width = xmax - xmin
    frame_height = ymax - ymin

    i_viz = np.matmul(viz_map, np.arange(viz_map.shape[0], dtype=np.int))
    # Handle the model's two internal features.
    i_viz = np.concatenate((np.array([0, 1], dtype=np.int), i_viz + 2))

    y_gap = radius
    n_image_rows = 4
    im_height = (frame_height - (n_image_rows + 1) * y_gap) / n_image_rows
    x_gap = (frame_width - 3 * im_height) / 3

    vt.scatter_2D(
        model.prefix_rewards[i_viz, :][:, i_viz],
        x0=xmin + 2 * x_gap + im_height,
        y0=ymin + (n_image_rows - 1) * y_gap + (n_image_rows - 1) * im_height,
        width=im_height,
        height=im_height,
        xlabel='goals',
        ylabel='features',
        title='Prefix rewards',
    )

    vt.scatter_2D(
        model.prefix_curiosities[i_viz, :][:, i_viz],
        x0=xmin + x_gap,
        y0=ymin + (n_image_rows - 1) * y_gap + (n_image_rows - 1) * im_height,
        width=im_height,
        height=im_height,
        xlabel='goals',
        ylabel='features',
        title='Prefix curiosities',
    )

    # TODO: weight all by feature activities
    sequences = model.sequence_occurrences / (model.prefix_occurrences + 1)
    # Goal direction is x.
    # Post feature direction is y.
    futures = np.max(sequences, axis=0)

    vt.scatter_2D(
        model.prefix_uncertainties[i_viz, :][:, i_viz],
        x0=xmin + x_gap,
        y0=ymin + (n_image_rows - 2) * y_gap + (n_image_rows - 2) * im_height,
        width=im_height,
        height=im_height,
        xlabel='goals',
        ylabel='features',
        title='Prefix uncertainties',
    )

    vt.scatter_2D(
        futures[i_viz, :][:, i_viz],
        x0=xmin + 2 * x_gap + im_height,
        y0=ymin + (n_image_rows - 2) * y_gap + (n_image_rows - 2) * im_height,
        width=im_height,
        height=im_height,
        xlabel='goals',
        ylabel='outcomes',
        title='Futures',
    )

    sequences_viz = np.moveaxis(
        sequences[i_viz, :, :][:, i_viz, :][:, :, i_viz],
        [0, 1, 2], [2, 1, 0])
    vt.scatter_3D(
        sequences_viz,
        x0=xmin + x_gap,
        y0=ymin + y_gap,
        width=2 * im_height + x_gap,
        height=2 * im_height + y_gap,
    )
