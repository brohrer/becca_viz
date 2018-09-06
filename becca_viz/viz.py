"""
Show the world and what's going on inside the brain.
"""
import os

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import becca_viz.viz_tools as vt
import becca_viz.preprocessor_viz as preprocessor_viz
import becca_viz.postprocessor_viz as postprocessor_viz
import becca_viz.input_filter_viz as input_filter_viz
import becca_viz.ziptie_viz as ziptie_viz
import becca_viz.featurizer_viz as featurizer_viz
import becca_viz.feature_filter_viz as feature_filter_viz
import becca_viz.model_viz as model_viz


lwf = .3  # Linewidth of the frames

wd_ = 18  # Total width of the image
ht_ = 9  # Total heigh of the image
brd = .75  # Border thickness around frames

halfwidth_weight = 5
centerwidth_weight = 1
total_weight = 2 * halfwidth_weight + centerwidth_weight

no_borders = wd_ - brd * 3  # Width available after borders are removed
wdh = no_borders * halfwidth_weight / total_weight  # Width of wide columns
wdc = no_borders * centerwidth_weight / total_weight  # Center column width

htf = ht_ - brd * 2  # Full height, minus borders
n_zipties = 2  # Assumed number of zipties
n_rows = 1 + 2 * n_zipties  # Total number of frames stacked vertically
no_borders_ht = ht_ - brd * (n_zipties + 2)  # Total non-border height
htr = no_borders_ht / n_rows  # height of each row
wdq = (wdh - brd) / 2  # Quarter-width
wds = (wdh - brd) / 3  # Sixth-width

rad = htr / 6  # The radius of the rounded corners on frames

preprocessor_bbox = [brd, brd + 2 * wds, brd, brd + htr]
postprocessor_bbox = [2 * brd + 2 * wds, brd + wdh, brd, brd + htr]
filt_0_bbox = [brd, brd + wdh, 2 * brd + htr, 2 * brd + 2 * htr]
ziptie_0_bbox = [brd, brd + wdh, 2 * brd + 2 * htr, 2 * brd + 3 * htr]
filt_1_bbox = [brd, brd + wdh, 3 * brd + 3 * htr, 3 * brd + 4 * htr]
ziptie_1_bbox = [brd, brd + wdh, 3 * brd + 4 * htr, 3 * brd + 5 * htr]
feature_filt_bbox = [2 * brd + wdh, 2 * brd + wdh + wdc, brd, brd + htf]
model_bbox = [
    2 * brd + wdh + wdc, 2 * brd + 2 * wdh + wdc, brd, brd + htf]


def visualize(brain):
    """
    Render the sensor information making its way up through the brain.

    Parameters
    ----------
    brain: Brain
    """
    # TODO: Incorporate this into the main visualization.
    brain.affect.visualize(brain)
    create_background()
    x_inputs, preprocessor_viz_map = preprocessor_viz.render(
        brain.preprocessor, preprocessor_bbox, radius=rad)
    x_commands, postprocessor_viz_map = postprocessor_viz.render(
        brain.postprocessor, postprocessor_bbox, radius=rad)
    n_pre = preprocessor_viz_map.shape[0]
    n_post = postprocessor_viz_map.shape[0]

    pool_0_viz_map = np.block([
        [np.zeros((n_post, n_pre), dtype=np.int), postprocessor_viz_map],
        [preprocessor_viz_map, np.zeros((n_pre, n_post), dtype=np.int)]])
    # TODO: handle multiple zipties
    x_pool_0 = np.concatenate((x_inputs, x_commands))
    x_cables_0, cables_0_viz_map = input_filter_viz.render(
        brain.featurizer.filter,
        filt_0_bbox,
        x_pool_0,
        pool_0_viz_map,
        preprocessor_bbox[3],  # max y value of the Preprocessor
        radius=rad)
    x_pool_1, pool_1_viz_map = ziptie_viz.render(
        brain.featurizer.ziptie,
        ziptie_0_bbox,
        x_cables_0,
        cables_0_viz_map,
        filt_0_bbox[3],  # max y value of the InputFilter
        radius=rad)
    pool_viz_map = [pool_0_viz_map, pool_1_viz_map]
    y_pool_feature, feature_pool_viz_map = featurizer_viz.render(
        brain.featurizer,
        feature_filt_bbox,
        pool_viz_map,
        radius=rad)
    y_feature, feature_viz_map = feature_filter_viz.render(
        brain.model.filter,
        feature_filt_bbox,
        y_pool_feature,
        feature_pool_viz_map,
        radius=rad)
    model_viz.render(
        brain.model,
        brain.actor,
        model_bbox,
        feature_viz_map,
        radius=rad)

    finalize(brain, dpi=300)


def labels(brain):
    """
    Generate a set of labels for the visualization.
    """
    create_background()
    title_frame("Preprocessor", preprocessor_bbox, radius=rad)
    title_frame("Postprocessor", postprocessor_bbox, radius=rad)
    title_frame("Cable filter 0", filt_0_bbox, radius=rad)
    title_frame("Ziptie 0", ziptie_0_bbox, radius=rad)
    title_frame("Cable filter 1", filt_1_bbox, radius=rad)
    title_frame("Ziptie 1", ziptie_1_bbox, radius=rad)
    title_frame("Feature\nfilter", feature_filt_bbox, radius=rad)
    title_frame("Model", model_bbox, radius=rad)
    model_viz.labels(model_bbox, radius=rad)

    finalize(brain, dpi=300, tag='labels')


def title_frame(title, bbox, radius=0):
    """
    Label the rendered components.
    """
    xmin, xmax, ymin, ymax = bbox
    plt.text(
        xmin + radius,
        ymax - radius,
        title,
        fontsize=16,
        color=vt.copper,
        verticalalignment="top",
        family="sans-serif",
    )


def create_background(edgecolor=vt.oxide, facecolor=vt.dark_grey):
    """
    Set up the backdrop for the visualization.
    """
    fig = plt.figure(num=84782, figsize=(wd_, ht_), frameon=False)
    fig.clf()
    ax = plt.gca()
    ax.add_patch(patches.Rectangle(
        (0, 0),
        wd_,
        ht_,
        facecolor=facecolor,
        edgecolor='none',
        zorder=-16.,
    ))
    # Preprocessor frame
    vt.draw_frame(
        bbox=preprocessor_bbox,
        radius=rad,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    # Postprocessor frame
    vt.draw_frame(
        bbox=postprocessor_bbox,
        radius=rad,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    # Cable filter 0 frame
    vt.draw_frame(
        bbox=filt_0_bbox,
        radius=rad,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    # Ziptie 0 frame
    vt.draw_frame(
        bbox=ziptie_0_bbox,
        radius=rad,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    # Cable filter 1 frame
    vt.draw_frame(
        bbox=filt_1_bbox,
        radius=rad,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    # Ziptie 1 frame
    vt.draw_frame(
        bbox=ziptie_1_bbox,
        radius=rad,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    # feature filter frame
    vt.draw_frame(
        bbox=feature_filt_bbox,
        radius=rad,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    # model frame
    vt.draw_frame(
        bbox=model_bbox,
        radius=rad,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )

    return


def finalize(brain, dpi=300, tag='activity', verbose=True):
    """
    Complete any final formatting and save a copy of the figure.

    Parameters
    ----------
    brain: Brain
    dpi: int
        The dots per inch for the saved figure.
    phase: int
        During each time step the brain does a lot and it's tough to get
        it all into one image. To handle this, it is broken up into phases.
        Phase 1 is sensing (an upward pass through the architecture) and
        Phase 2 is acting ( a downward pass through the architecture).

    """
    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')

    filename = 'becca_{name}_{dpi:04d}_{age:08d}_{tag}.png'.format(
        name=brain.name, age=brain.timestep, dpi=dpi, tag=tag)
    pathname = os.path.join(brain.log_dir, filename)
    plt.savefig(pathname, format='png', dpi=dpi)
    if verbose:
        print('Saved', pathname)
