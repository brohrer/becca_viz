import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.patches as patches

# Colors for plotting
dark_grey = (0.05, 0.05, 0.05)
light_grey = (0.9, 0.9, 0.9)
red = (0.9, 0.3, 0.3)
# Becca pallette
copper_highlight = (253/255, 249/255, 240/255)
light_copper = (242/255, 166/255, 108/255)
copper = (175/255, 102/255, 53/255)
dark_copper = (132/255, 73/255, 36/255)
copper_shadow = (25/255, 22/255, 20/255)
oxide = (20/255, 120/255, 150/255)

cmap_1 = colors.LinearSegmentedColormap.from_list(
    "", ["black", light_copper])
cmap_2 = colors.LinearSegmentedColormap.from_list(
    "", [oxide, "black", light_copper])

xlabel_shift = 3 / 4
ylabel_shift = 1 / 2
title_shift = 1 / 100
text_shift = 1 / 20


def draw_frame(
    bbox=None,
    radius=.01,
    edgecolor=copper,
    facecolor=None,
    linewidth=1,
):
    """
    Draw a frame with rounded corners.

    Parameters
    ----------
    bbox: list of floats
        Of the form [x_left, x_right, y_bottom, y_top]
    radius: float
        The radius of the rounded corners
    edgecolor, facecolor: pyplot color specifier
    linewidth: float
        Thickness of border line.
    """
    ax = plt.gca()
    xmin, xmax, ymin, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    z_frame = -8
    z_fill = -10

    # Fill in the background.
    if facecolor is not None:
        # Fill the tall part of the frame.
        ax.add_patch(patches.Rectangle(
            (xmin + radius, ymin),
            width - 2 * radius,
            height,
            facecolor=facecolor,
            edgecolor='none',
            zorder=z_fill,
        ))
        # Fill the wide part of the frame.
        ax.add_patch(patches.Rectangle(
            (xmin, ymin + radius),
            width,
            height - 2 * radius,
            facecolor=facecolor,
            edgecolor='none',
            zorder=z_fill,
        ))
        # Fill the corners.
        ax.add_patch(patches.Circle(
            (xmin + radius, ymin + radius),
            radius=radius,
            facecolor=facecolor,
            edgecolor='none',
            zorder=z_fill,
        ))
        ax.add_patch(patches.Circle(
            (xmin + radius, ymax - radius),
            radius=radius,
            facecolor=facecolor,
            edgecolor='none',
            zorder=z_fill,
        ))
        ax.add_patch(patches.Circle(
            (xmax - radius, ymin + radius),
            radius=radius,
            facecolor=facecolor,
            edgecolor='none',
            zorder=z_fill,
        ))
        ax.add_patch(patches.Circle(
            (xmax - radius, ymax - radius),
            radius=radius,
            facecolor=facecolor,
            edgecolor='none',
            zorder=z_fill,
        ))

    # Draw the frame.
    # Right wall
    plt.plot(
        [xmin + radius, xmax - radius],
        [ymax, ymax],
        color=edgecolor,
        linewidth=linewidth,
        zorder=z_frame,
    )
    # Left wall
    plt.plot(
        [xmin + radius, xmax - radius],
        [ymin, ymin],
        color=edgecolor,
        linewidth=linewidth,
        zorder=z_frame,
    )
    # Bottom wall
    plt.plot(
        [xmin, xmin],
        [ymin + radius, ymax - radius],
        color=edgecolor,
        linewidth=linewidth,
        zorder=z_frame,
    )
    # Top wall
    plt.plot(
        [xmax, xmax],
        [ymin + radius, ymax - radius],
        color=edgecolor,
        linewidth=linewidth,
        zorder=z_frame,
    )
    # Upper right corner
    plot_arc(
        color=edgecolor,
        linewidth=linewidth,
        radius=radius,
        theta_0=0,
        theta_1=np.pi / 2,
        x_center=xmax - radius,
        y_center=ymax - radius,
        zorder=z_frame,
    )
    # Upper left corner
    plot_arc(
        color=edgecolor,
        linewidth=linewidth,
        radius=radius,
        theta_0=np.pi / 2,
        theta_1=np.pi,
        x_center=xmin + radius,
        y_center=ymax - radius,
        zorder=z_frame,
    )
    # Lower right corner
    plot_arc(
        color=edgecolor,
        linewidth=linewidth,
        radius=radius,
        theta_0=3 * np.pi / 2,
        theta_1=2 * np.pi,
        x_center=xmax - radius,
        y_center=ymin + radius,
        zorder=z_frame,
    )
    # Lower left corner
    plot_arc(
        color=edgecolor,
        linewidth=linewidth,
        radius=radius,
        theta_0=np.pi,
        theta_1=3 * np.pi / 2,
        x_center=xmin + radius,
        y_center=ymin + radius,
        zorder=z_frame,
    )


def plot_arc(
    color='green',
    linewidth=1,
    radius=1,
    theta_0=0,
    theta_1=np.pi/2,
    x_center=0,
    y_center=0,
    zorder=0,
):
    """
    Put an arc in the current axes.

    Parameters
    ----------
    radius: float
    theta_0, theta_1: floats
        The start and end angles of the arc, in radians.
        theta_0 must be smaller than theta_1
    x_center, y_center: floats
        The coordinates of the center of the arc.
    z_order: int
        The over/under-lap priority for the arc.
    """
    ds = .001
    # dtheta = ds / radius
    theta = np.arange(theta_0, theta_1, ds)
    x = x_center + np.cos(theta) * radius
    y = y_center + np.sin(theta) * radius

    plt.plot(x, y, color=color, linewidth=linewidth, zorder=zorder)


def plot_point_activity(
    x, y,
    activity,
    spacing,
    activity_color=light_copper,
):
    """
    Draw a point that represents the activity of a signal it carries.
    """
    r = np.minimum(np.maximum(.01, spacing / 24.), .03)
    circle = plt.Circle(
        (x, y),
        r,
        color=copper_shadow,
        zorder=6.,
    )
    plt.gca().add_artist(circle)
    r_activity = activity * r * 2.
    circle = plt.Circle(
        (x, y),
        r_activity,
        color=activity_color,
        zorder=7. + activity,
    )
    plt.gca().add_artist(circle)


def plot_curve_activity(
    x_start, x_end, y_start, y_end,
    activity,
    shadow_color=copper_shadow,
    activity_color=light_copper,
):
    """
    Draw a smooth curve connecting two points.
    """
    t = np.arange(0., np.pi, .01)
    curve = np.cos(t)
    offset = (x_start + x_end) / 2.
    dilation = (y_end - y_start) / np.pi
    scale = (
        np.sign(y_start - y_end) *
        (x_end - x_start) / 2.
    )
    plt.plot(
        offset + curve * scale,
        y_start + t * dilation,
        color=shadow_color,
        linewidth=.8,
        zorder=2.,
    )
    plt.plot(
        offset + curve * scale,
        y_start + t * dilation,
        color=activity_color,
        linewidth=activity,
        zorder=activity + 2.,
    )


def plot_curve_activity_horiz(
    y_start, y_end, x_start, x_end,
    activity,
    shadow_color=copper_shadow,
    activity_color=light_copper,
):
    """
    Draw a smooth curve connecting two points.
    """
    t = np.arange(0., np.pi, .01)
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
        color=shadow_color,
        linewidth=.8,
        zorder=2.,
    )
    plt.plot(
        x_start + t * dilation,
        offset + curve * scale,
        color=activity_color,
        linewidth=activity,
        zorder=activity + 2.,
    )


def plot_line_activity(x, y, activity):
    """
    Draw a line that represents the activity of a signal it carries.
    """

    plt.plot(
        x, y,
        color=copper_shadow,
        linewidth=.8,
        zorder=2.,
    )
    plt.plot(
        x, y,
        color=light_copper,
        linewidth=activity,
        zorder=activity + 2.,
    )


def scatter(x, y, c):
    cb = c.copy()
    cb[:, 3] *= cb[:, 3]
    cc = cb.copy()
    cc[:, 3] *= cc[:, 3]
    plt.scatter(x, y, c="black", marker='.', s=15, edgecolor='none', zorder=1)
    plt.scatter(x, y, c=cc, marker='.', s=7, edgecolor='none', zorder=2)
    plt.scatter(x, y, c=cb, marker='.', s=4, edgecolor='none', zorder=3)
    plt.scatter(x, y, c=c, marker='.', s=2, edgecolor='none', zorder=4)


def scatter_1D(
    arr,
    x0=0,
    y0=0,
    width=1,
    xlabel='',
    title='',
):
    """
    Make a 1D scatter plot.

    Put it in a row.
    """
    npts = arr.size
    x = np.zeros(npts)
    y = np.zeros(npts)
    c = np.zeros((npts, 4))
    c[:, :3] = np.array(light_copper)
    x_unit = width / (npts - 1)

    for i_x in range(npts):
        x[i_x] = i_x * x_unit
        val = np.maximum(np.minimum(arr[i_x], 1), -1)
        if val > 0:
            c[i_x, 3] = val
        else:
            c[i_x, :3] = np.array(oxide)
            c[i_x, 3] = -val

    x += x0
    y += y0
    scatter(x, y, c)

    text_sep = width * text_shift
    plt.text(
        x0 + width * xlabel_shift,
        y0 - text_sep,
        xlabel,
        fontsize=4,
        color=copper,
        verticalalignment="top",
        horizontalalignment="left",
        family="sans-serif",
    )
    plt.text(
        x0 + width * title_shift,
        y0 + text_sep,
        title,
        fontsize=6,
        color=copper,
        verticalalignment="bottom",
        horizontalalignment="left",
        family="sans-serif",
    )
    return


def scatter_2D(
    arr,
    x0=0,
    y0=0,
    width=1,
    height=1,
    xlabel='',
    ylabel='',
    title='',
):
    """
    Make a 2D scatter plot.

    row number corresponds to y value
    column number corresponds to x value
    """
    nrows, ncols = arr.shape
    npts = nrows * ncols
    x = np.zeros(npts)
    y = np.zeros(npts)
    c = np.zeros((npts, 4))
    c[:, :3] = np.array(light_copper)
    y_unit = height / (nrows - 1)
    x_unit = width / (ncols - 1)

    for i_x in range(ncols):
        for i_y in np.arange(nrows - 1, -1, -1, dtype=np.int):
                j = i_x * nrows + i_y
                x[j] = i_x * x_unit
                y[j] = i_y * y_unit
                val = np.maximum(np.minimum(arr[i_y, i_x], 1), -1)
                if val > 0:
                    c[j, 3] = val
                else:
                    c[j, :3] = np.array(oxide)
                    c[j, 3] = -val

    x += x0
    y += y0
    scatter(x, y, c)

    text_sep = np.minimum(height, width) * text_shift
    plt.text(
        x0 + width * xlabel_shift,
        y0 - text_sep,
        xlabel,
        fontsize=4,
        color=copper,
        verticalalignment="top",
        horizontalalignment="left",
        family="sans-serif",
    )
    plt.text(
        x0 - text_sep,
        y0 + height * ylabel_shift,
        ylabel,
        fontsize=4,
        color=copper,
        rotation=-90,
        verticalalignment="center",
        horizontalalignment="right",
        family="sans-serif",
    )
    plt.text(
        x0 + width * title_shift,
        y0 + height + text_sep,
        title,
        fontsize=6,
        color=copper,
        verticalalignment="bottom",
        horizontalalignment="left",
        family="sans-serif",
    )
    return


def scatter_3D(
    arr,
    x0=0,
    y0=0,
    width=1,
    height=1,
    xlabel='goals',
    ylabel='features',
    zlabel='outcomes',
    title='Active sequences',
):
    """
    """
    nrows, ncols, ndeps = arr.shape
    npts = nrows * ncols * ndeps
    x = np.zeros(npts)
    y = np.zeros(npts)
    c = np.zeros((npts, 4))
    c[:, :3] = np.array(light_copper)

    center = np.array([x0 + width / 2, y0 + height / 2])
    # Direction vectors with foreshortening and scaling
    fs = [.9, .7, 1]
    scale = .6
    length = np.sqrt(width * height)
    # theta
    x_dir = fs[0] * scale * np.array([
        np.cos(-10 * np.pi / 180), np.sin(-10 * np.pi / 180)])
    y_dir = fs[1] * scale * np.array([
        np.cos(30 * np.pi / 180), np.sin(30 * np.pi / 180)])
    z_dir = fs[2] * scale * np.array([0, 1])
    # Unit vectors
    x_unit = x_dir * length / ncols
    y_unit = y_dir * length / nrows
    z_unit = z_dir * length / ndeps
    # Origin
    x_0 = -x_unit * ncols / 2
    y_0 = -y_unit * nrows / 2
    z_0 = -z_unit * ndeps / 2
    xy_0 = center + x_0 + y_0 + z_0

    x = np.zeros(npts)
    y = np.zeros(npts)
    c = np.zeros((npts, 4))
    c[:, :3] = np.array(light_copper)

    for i_x in range(ncols):
        for i_y in np.arange(nrows - 1, -1, -1, dtype=np.int):
            for i_z in range(ndeps):
                j = i_x * nrows * ndeps + i_y * ndeps + i_z
                x_part = i_x * x_unit
                y_part = i_y * y_unit
                z_part = i_z * z_unit
                x[j], y[j] = x_part + y_part + z_part
                c[j, 3] = np.minimum(arr[i_y, i_x, i_z], 1)
    x += xy_0[0]
    y += xy_0[1]
    scatter(x, y, c)

    plt.text(
        x0 + .22 * width,
        y0 + .08 * height,
        xlabel,
        fontsize=4,
        color=copper,
        rotation=-10,
        verticalalignment="top",
        horizontalalignment="left",
        family="sans-serif",
    )
    plt.text(
        x0 + .05 * width,
        y0 + .4 * height,
        ylabel,
        fontsize=4,
        color=copper,
        rotation=-90,
        verticalalignment="center",
        horizontalalignment="right",
        family="sans-serif",
    )
    plt.text(
        x0 + .7 * width,
        y0 + .15 * height,
        zlabel,
        fontsize=4,
        color=copper,
        rotation=30,
        verticalalignment="top",
        horizontalalignment="left",
        family="sans-serif",
    )
    plt.text(
        x0 + width * title_shift,
        y0 + height * .86,
        title,
        fontsize=6,
        color=copper,
        verticalalignment="bottom",
        horizontalalignment="left",
        family="sans-serif",
    )
    return


def nd_map(arr, map):
    """
    Apply a map to a multidimensional array.

    Parameters
    ----------
    arr: nD array of floats
    map: 2D array of ints

    Returns
    -------
    nD array of floats
        Re-mapped
    """
    nD = len(arr.shape)
    dim_list = list(np.arange(nD, dtype=np.int))
    shift_list = list(dim_list)
    shift_list.append(shift_list.pop(0))

    map_size = map.shape[0]
    if nD == 1:
        arr = arr[:map_size]
    elif nD == 2:
        arr = arr[:map_size, :map_size]
    elif nD == 3:
        arr = arr[:map_size, :map_size, :map_size]
    elif nD == 4:
        arr = arr[:map_size, :map_size, :map_size, :map_size]

    for _ in range(nD):
        arr = np.matmul(arr, map)
        arr = np.moveaxis(arr, dim_list, shift_list)

    return arr
