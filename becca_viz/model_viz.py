import matplotlib.pyplot as plt
import numpy as np

import becca_viz.viz_tools as vt


n_image_rows = 4
n_image_cols = 3


def render(model, actor, bbox, viz_map, radius=0):
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
    autoscaled_viz_rewards = viz_rewards / (
        np.max(np.abs(viz_rewards)) + 1e-2)
    vt.scatter_2D(
        autoscaled_viz_rewards,
        x0=xmin + 2 * x_gap + im_height,
        y0=(ymin + (n_image_rows - 1) * y_gap
            + (n_image_rows - 1) * im_height),
        width=im_height,
        height=im_height,
        xlabel='goals',
        ylabel='features',
        title='Active rewards',
    )

    curiosities = (model.feature_activities[:, np.newaxis]
                   * model.prefix_curiosities)[2:, 2:]
    viz_curiosities = vt.nd_map(curiosities, viz_map)
    autoscaled_viz_curiosities = viz_curiosities / (
        np.max(viz_curiosities) + 1e-2)
    vt.scatter_2D(
        autoscaled_viz_curiosities,
        x0=xmin + x_gap,
        y0=(ymin + (n_image_rows - 1) * y_gap
            + (n_image_rows - 1) * im_height),
        width=im_height,
        height=im_height,
        xlabel='goals',
        ylabel='features',
        title='Active curiosities',
    )

    # TODO: weight all by feature activities
    sequences = (model.feature_activities[:, np.newaxis, np.newaxis] *
                 model.sequence_likelihoods)
    sequences_viz = vt.nd_map(sequences[2:, 2:, 2:], viz_map)
    # Goal direction is x.
    # Post feature direction is y.
    viz_futures = np.max(sequences_viz, axis=0).transpose()
    # sequences = np.moveaxis(sequences, [0, 1, 2], [1, 2, 0])

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

    conditional_rewards_viz = vt.nd_map(
        model.conditional_rewards[2:], viz_map)
    vt.scatter_1D(
        conditional_rewards_viz,
        x0=xmin + 3 * x_gap + 2 * im_height,
        y0=ymin + 4 * y_gap + 3.5 * im_height,
        width=im_height,
        xlabel='goals',
        title='Conditional rewards',
    )

    conditional_curiosities_viz = vt.nd_map(
        model.conditional_curiosities[2:], viz_map)
    vt.scatter_1D(
        conditional_curiosities_viz,
        x0=xmin + 3 * x_gap + 2 * im_height,
        y0=ymin + 3 * y_gap + 2.5 * im_height,
        width=im_height,
        xlabel='goals',
        title='Conditional curiosities',
    )

    conditional_goal_rewards = np.max(
        model.conditional_predictions *
        model.goal_activities[np.newaxis, :], axis=1)
    conditional_goal_rewards_viz = vt.nd_map(
        conditional_goal_rewards[2:], viz_map)
    vt.scatter_1D(
        conditional_goal_rewards_viz,
        x0=xmin + 3 * x_gap + 2 * im_height,
        y0=ymin + 1 * y_gap + 0.5 * im_height,
        width=im_height,
        xlabel='goals',
        title='Conditional goal rewards',
    )

    goal_activities_viz = vt.nd_map(
        actor.goal_collection[2:], viz_map)
    # model.goal_activities[2:], viz_map)
    vt.scatter_1D(
        goal_activities_viz,
        x0=xmin + 3 * x_gap + 2 * im_height,
        y0=ymin + 2 * y_gap + 1.5 * im_height,
        width=im_height,
        xlabel='goals',
        title='Goal collection',
    )
    return


def labels(bbox, radius=0):
    """
    Add model-specific labels to the labeled visualization.
    """
    xmin, xmax, ymin, ymax = bbox
    frame_width = xmax - xmin
    frame_height = ymax - ymin

    y_gap = radius
    im_height = (frame_height - (n_image_rows + 1) * y_gap) / n_image_rows
    x_gap = (frame_width - n_image_cols * im_height) / (n_image_cols + 1)

    label_text(
        text='Active\nreward\nprefixes',
        x=xmin + 2 * x_gap + im_height,
        y=ymin + (n_image_rows - 0) * y_gap + (n_image_rows - .5) * im_height,
    )

    label_text(
        text='Active\ncuriosity\nprefixes',
        x=xmin + x_gap,
        y=ymin + (n_image_rows - 0) * y_gap + (n_image_rows - .5) * im_height,
    )

    label_text(
        text='Prefix\nactivities',
        x=xmin + x_gap,
        y=ymin + (n_image_rows - 1) * y_gap + (n_image_rows - 1.5) * im_height,
    )

    label_text(
        text='Active\nfutures',
        x=xmin + 2 * x_gap + im_height,
        y=ymin + (n_image_rows - 1) * y_gap + (n_image_rows - 1.5) * im_height,
    )

    label_text(
        text='Active\nsequences',
        x=xmin + x_gap,
        y=ymin + 2 * y_gap + im_height,
    )

    label_text(
        text='Conditional\nrewards',
        x=xmin + 3 * x_gap + 2 * im_height,
        y=ymin + 4 * y_gap + 3.5 * im_height,
    )

    label_text(
        text='Conditional\ncuriosities',
        x=xmin + 3 * x_gap + 2 * im_height,
        y=ymin + 3 * y_gap + 2.5 * im_height,
    )

    label_text(
        text='Goal\nactivities',
        x=xmin + 3 * x_gap + 2 * im_height,
        y=ymin + 2 * y_gap + 1.5 * im_height,
    )

    label_text(
        text='Goal\nrewards',
        x=xmin + 3 * x_gap + 2 * im_height,
        y=ymin + 1 * y_gap + 0.5 * im_height,
    )


def label_text(text='', x=0, y=0):
    """
    Craft the label text.
    """
    plt.text(
        x,
        y,
        text,
        fontsize=12,
        color=vt.copper,
        verticalalignment="center",
        family="sans-serif",
    )


def render_state(model, bbox, viz_map, radius=0):
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
