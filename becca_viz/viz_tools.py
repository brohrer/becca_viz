import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Colors for plotting
dark_grey = (0.05, 0.05, 0.05)
light_grey = (0.9, 0.9, 0.9)
red = (0.9, 0.3, 0.3)
# Becca pallette
copper_highlight = (253./255., 249./255., 240./255.)
light_copper = (242./255., 166./255., 108./255.)
copper = (175./255., 102./255, 53./255.)
dark_copper = (132./255., 73./255., 36./255.)
copper_shadow = (25./255., 22./255, 20./255.)
oxide = (20./255., 120./255., 150./255.)


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
        theta_1= 3 * np.pi / 2,
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
    dtheta = ds / radius
    theta = np.arange(theta_0, theta_1, ds)
    x = x_center + np.cos(theta) * radius
    y = y_center + np.sin(theta) * radius
    
    plt.plot(x, y, color=color, linewidth=linewidth, zorder=zorder)


def plot_point_activity(x, y, activity, spacing):
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
        color=light_copper,
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
    t = np.arange(0., np.pi , .01)
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

