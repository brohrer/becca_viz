import os

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import becca_viz.viz_tools as vt


def visualize(affect, brain, verbose=True):
    """
    Parameters
    ----------
    affect: Affect
    brain: Brain
    """
    # Plot the lifetime record of the reward.
    fig = plt.figure(11111)
    fig.clf()
    ax = plt.gca()

    color = (np.array(vt.copper) +
             np.random.normal(size=3, scale=.1))
    color = np.maximum(np.minimum(color, 1), 0)
    color = tuple(color)
    linewidth = np.random.normal(loc=2.5)
    linewidth = 2
    linewidth = np.maximum(1, linewidth)
    plt.plot(
        np.array(affect.reward_steps) / 1000.,
        affect.reward_history,
        color=color,
        linewidth=linewidth,
    )

    x_min = 0
    x_max = affect.reward_steps[-1] / 1000
    y_min = np.minimum(np.min(affect.reward_history), 0)
    y_max = np.maximum(np.max(affect.reward_history), 1)
    ax.add_patch(patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        facecolor=vt.copper_highlight,
        edgecolor='none',
        zorder=-16,
    ))
    plt.axis([x_min, x_max, y_min, y_max])

    plt.xlabel('Thousands of time steps')
    plt.ylabel('Average reward')
    plt.title('Reward history for {0}'.format(brain.name))

    # Save a copy of the plot.
    filename = 'reward_history_{0}.png'.format(brain.name)
    pathname = os.path.join(brain.log_dir, filename)
    plt.savefig(pathname, format='png')
    if verbose:
        print('Saved', pathname)
    return
