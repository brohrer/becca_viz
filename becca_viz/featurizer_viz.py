"""
Visualize the featurizer for reporting and debugging.
"""

from __future__ import print_function
import os

from mpl_toolkits.axes_grid1 import ImageGrid
#import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def visualize(featurizer, brain, world):
    """
    Show the current state of the featurizer.

    Parameters
    ----------
    brain : Brain
        The number of actions in the brain is referenced
        to customize the display.
    featurizer : Featurizer
        The featurizer being visualized.
    """
    # activity_threshold : the level at which we can ignore
    # an element's activity in order to simplify display.
    activity_threshold = .01
    print(featurizer.name)

    print("Input activities")
    for i_input, activity in enumerate(featurizer.input_activities):
        if activity > activity_threshold:
            print(" ".join(["input", str(i_input), ":",
                            "activity ", str(activity)]))

    N = int(np.ceil(featurizer.max_num_features ** .5))
    M = int(np.ceil(featurizer.max_num_features / float(N)))
    grid_shape = (N, M)

    # fig : matplotlob Figure
    #     A single figure that summarizes all the features
    #     that have been created.
    fignum = 85673
    # Allow a 9 x 9 in plot for each feature.
    # This helps line thicknesses to render properly.
    fig = plt.figure(fignum, grid_shape)
    plt.clf()
    # ax_grid : list of matplotlib Axis
    #     All the axes in the visualization figure.
    ax_grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=grid_shape, # creates an NxM grid of axes
        axes_pad=0.1,  # pad between axes in inches.
        )
    for axis in ax_grid:
        plt.sca(axis)
        plt.axis("off")
    try:
        i_feature = 0

        # Render the features.
        for i_feature in range(featurizer.max_num_features):
            feature_activities = np.zeros(featurizer.max_num_features)
            feature_activities[i_feature] = 1.
            input_activities = featurizer.defeaturize(feature_activities)
            if np.where(input_activities > 0.)[0].size > 0:
                actions = input_activities[:brain.num_actions]
                sensors = input_activities[brain.num_actions:]
                plt.sca(ax_grid[i_feature])
                plt.cla()
                world.render_sensors_actions(sensors, actions)
            i_feature += 1

        fig.show()
        fig.canvas.draw()

        # Save a copy of the plot.
        filename = 'featurizer_features_{0}.png'.format(brain.name)
        pathname = os.path.join(brain.log_dir, filename)
        plt.figure(fignum)
        plt.savefig(pathname, format='png', dpi=600)

    except:
        print('Featurizer failed to render features.')
        featurizer.ziptie.visualize()
