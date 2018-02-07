"""
Visualization for the model.
"""

from __future__ import print_function
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def set_up_visualization(model, brain):
    """
    Initialize the visualization of the model.

    To make the visualization interpretable, there are some
    annotations and visual guides added.

    Parameters
    ----------
    brain : Brain
        The number of actions in the brain is referenced
        to customize the display.
    model : Model
        The model being visualized.
    """
    # Prepare visualization.
    plt.bone()
    # fig : matplotlib figure
    #     The figure in which a visual representation of the results
    #     will be presented.
    # ax_curiosities,
    # ax_rewards,
    # ax_ocurrences : matplotlib axes
    #     The axes in which each of these 2D arrays will be rendered
    #     as images.
    plt.figure(num=73857, figsize=(9, 9))
    plt.clf()
    model.fig, (
        (model.ax_rewards, model.ax_curiosities),
        (model.ax_activities, model.ax_occurrences)) = (
            plt.subplots(2, 2, num=73857))

    def dress_axes(ax):
        """
        Decorate the axes appropriately with visual cues.
        """
        plt.sca(ax)
        ax.add_patch(patches.Rectangle(
            (-.5, - .5),
            model.num_features,
            2.,
            facecolor='green',
            edgecolor='none',
            alpha=.16))
        ax.add_patch(patches.Rectangle(
            (-.5, -.5),
            2.,
            model.num_features,
            facecolor='green',
            edgecolor='none',
            alpha=.16))
        ax.add_patch(patches.Rectangle(
            (-.5, brain.num_actions + 2. -.5),
            model.num_features,
            brain.num_sensors,
            facecolor='green',
            edgecolor='none',
            alpha=.16))
        ax.add_patch(patches.Rectangle(
            (brain.num_actions + 2. -.5, -.5),
            brain.num_sensors,
            model.num_features,
            facecolor='green',
            edgecolor='none',
            alpha=.16))
        ax.plot(
            [-.5, model.num_features - .5],
            [2. - .5, 2. - .5],
            color='blue',
            linewidth=.2)
        ax.plot(
            [2. - .5, 2. - .5],
            [-.5, model.num_features - .5],
            color='blue',
            linewidth=.2)
        ax.plot(
            [-.5, model.num_features - .5],
            [brain.num_actions + 2. - .5, brain.num_actions + 2. - .5],
            color='blue',
            linewidth=.2)
        ax.plot(
            [brain.num_actions + 2. - .5, brain.num_actions + 2. - .5],
            [-.5, model.num_features - .5],
            color='blue',
            linewidth=.2)
        ax.plot(
            [-.5, model.num_features - .5],
            [brain.num_sensors + brain.num_actions + 2. - .5,
             brain.num_sensors + brain.num_actions + 2. - .5],
            color='blue',
            linewidth=.2)
        ax.plot(
            [brain.num_sensors + brain.num_actions + 2. - .5,
             brain.num_sensors + brain.num_actions + 2. - .5],
            [-.5, model.num_features - .5],
            color='blue',
            linewidth=.2)
        plt.xlim([-.5, model.num_features - .5])
        plt.ylim([-.5, model.num_features - .5])
        ax.invert_yaxis()

    dress_axes(model.ax_rewards)
    dress_axes(model.ax_curiosities)
    dress_axes(model.ax_activities)
    dress_axes(model.ax_occurrences)


def visualize(model, brain):
    """
    Make a picture of the model.

    Parameters
    ----------
    brain : Brain
        The brain that this model belongs to.
    model : Model
        The model being visualized.
    """
    # Show prefix_rewards.
    ax = model.ax_rewards
    ax.imshow(
        model.prefix_rewards, vmin=-1., vmax=1., interpolation='nearest')
    ax.set_title('Rewards')
    ax.set_ylabel('Features')

    # Show prefix_curiosities.
    ax = model.ax_curiosities
    ax.imshow(
        model.prefix_curiosities, vmin=0., vmax=1., interpolation='nearest')
    ax.set_title('Curiosities')
    ax.set_xlabel('Goals')

    # Show prefix_activities.
    ax = model.ax_activities
    ax.imshow(
        model.prefix_activities,
        #model.prefix_credit,
        vmin=0.,
        vmax=1.,
        interpolation='nearest')
    ax.set_title('Activities')
    ax.set_xlabel('Goals')
    ax.set_ylabel('Features')

    # Show prefix_occurrences.
    ax = model.ax_occurrences
    log_occurrences = np.log10(model.prefix_occurrences + 1.)
    ax.imshow(log_occurrences, interpolation='nearest')
    ax.set_title('Occurrences, max = {0}'.format(
        int(10 ** np.max(log_occurrences))))
    ax.set_xlabel('Goals')

    model.fig.show()
    model.fig.canvas.draw()

    # Save a copy of the plot.
    filename = 'model_history_{0}.png'.format(brain.name)
    pathname = os.path.join(brain.log_dir, filename)
    plt.figure(73857)
    plt.savefig(pathname, format='png', dpi=300)
    return
