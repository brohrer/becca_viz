import numpy as np

import becca_viz.viz_tools as vt


n_image_rows = 4
n_image_cols = 2


def render(model, bbox, viz_map, max_floor=1e-2, radius=0):
    """
    Turn it into a picture.
    """
    xmin, xmax, ymin, ymax = bbox
    frame_width = xmax - xmin
    frame_height = ymax - ymin

    y_gap = 2 * radius
    im_height = (frame_height - (n_image_rows + 1) * y_gap) / n_image_rows
    x_gap = (frame_width - n_image_cols * im_height) / (n_image_cols + 1)

    rewards = (model.feature_activities[:, np.newaxis] *
               model.prefix_rewards)[2:, 2:]
    viz_rewards = vt.nd_map(rewards, viz_map)
    title = "Active rewards"
    vt.scatter_2D(
        viz_rewards,
        x0=xmin + 2 * x_gap + im_height,
        y0=(ymin + (n_image_rows - 1) * y_gap
            + (n_image_rows - 1) * im_height),
        width=im_height,
        height=im_height,
        xlabel='goals',
        ylabel='features',
        title=title,
        autoscale=True,
    )

    curiosities = (model.feature_activities[:, np.newaxis]
                   * model.prefix_curiosities)[2:, 2:]
    viz_curiosities = vt.nd_map(curiosities, viz_map)
    title = "Active curiosities"
    vt.scatter_2D(
        viz_curiosities,
        x0=xmin + x_gap,
        y0=(ymin + (n_image_rows - 1) * y_gap
            + (n_image_rows - 1) * im_height),
        width=im_height,
        height=im_height,
        xlabel='goals',
        ylabel='features',
        title=title,
        autoscale=True,
    )

    sequences = (model.feature_activities[:, np.newaxis, np.newaxis] *
                 model.sequence_likelihoods)
    sequences = np.moveaxis(sequences, [0, 1, 2], [2, 1, 0])
    sequences_viz = vt.nd_map(sequences[2:, 2:, 2:], viz_map)
    # Goal direction is x.
    # Post feature direction is y.
    viz_futures = np.max(sequences_viz, axis=2)

    prefix_activities = vt.nd_map(model.prefix_activities[2:, 2:], viz_map)
    vt.scatter_2D(
        prefix_activities,
        x0=xmin + x_gap,
        y0=(ymin + (n_image_rows - 2) * y_gap
            + (n_image_rows - 2) * im_height),
        width=im_height,
        height=im_height,
        xlabel='goals',
        ylabel='features',
        title='Prefix activities',
        autoscale=True,
    )

    vt.scatter_2D(
        viz_futures,
        x0=xmin + 2 * x_gap + im_height,
        y0=(ymin + (n_image_rows - 2) * y_gap
            + (n_image_rows - 2) * im_height),
        width=im_height,
        height=im_height,
        xlabel='goals',
        ylabel='outcomes',
        title='Futures',
        autoscale=True,
    )

    vt.scatter_3D(
        sequences_viz,
        x0=xmin + x_gap,
        y0=ymin + y_gap,
        width=2 * im_height + x_gap,
        height=2 * im_height + y_gap,
        xlabel='goals',
        ylabel='features',
        zlabel='outcomes',
        title='Active\nsequences',
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
