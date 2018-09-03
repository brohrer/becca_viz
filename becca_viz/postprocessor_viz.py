"""
Show what's going on inside the postprocessor.
"""
import numpy as np

import becca_viz.viz_tools as vt


def render(postprocessor, bbox, radius=0, phase=0):
    """
    Make a picture of the discretization that happens in the Preprocessor.

    Parameters
    ----------
    bbox: list of floats
        Of the form [x_bottom, x_top, y_left, y_right]
    phase: int
    postprocessor: Postprocessor
    radius: float
        The corner radius of the box.

    Returns
    -------
    x_commands: array of floats
        The absolute x positions of the command nodes.
    i_to_viz: 2D array of ints
        Mapping from commands to visualization order.
    """

    xmin, xmax, ymin, ymax = bbox
    frame_width = xmax - xmin
    # frame_height = ymax - ymin
    n_commands = postprocessor.n_commands
    n_actions = postprocessor.n_actions
    # Collect positions from all the commands.
    n_commands_per_action = int(n_commands / n_actions)
    x_command_spacing = (
        (frame_width - 2 * radius) / (n_commands + 1))
    x_commands = (xmin + radius + x_command_spacing *
                  np.cumsum(np.ones(n_commands)))
    x_action_spacing = (frame_width - 2 * radius) / (n_actions + 1)
    x_actions = (xmin + radius + x_action_spacing *
                 np.cumsum(np.ones(n_commands)))

    i_to_viz = np.eye(n_commands, dtype=np.int)

    # Show the upward, sensed commands from the previous time step.
    for i_action, _ in enumerate(postprocessor.actions):
        for j in range(n_commands_per_action):
            i_command = i_action * n_commands_per_action + j
            command_activity = postprocessor.previous_commands[
                i_command]

            vt.plot_point_activity(
                x_commands[i_command],
                ymax,
                command_activity,
                x_command_spacing,
            )
            vt.plot_curve_activity(
                x_actions[i_action],
                x_commands[i_command],
                ymin,
                ymax,
                command_activity,
            )

    offset = (x_actions[1] - x_actions[0]) / 10
    # Show the downward commands, to be executed during the next time step.
    for i_action, action in enumerate(postprocessor.actions):
        for j in range(n_commands_per_action):
            i_command = i_action * n_commands_per_action + j
            command_activity = postprocessor.command_activities[
                i_command]
            if command_activity > .02:
                vt.plot_point_activity(
                    x_commands[i_command] + offset,
                    ymax,
                    command_activity,
                    x_command_spacing,
                    activity_color=vt.oxide,
                )
                vt.plot_curve_activity(
                    x_actions[i_action] + offset,
                    x_commands[i_command] + offset,
                    ymin,
                    ymax,
                    command_activity,
                    activity_color=vt.oxide,
                )
                vt.plot_point_activity(
                    x_actions[i_action] + offset,
                    ymin,
                    action,
                    x_action_spacing,
                    activity_color=vt.oxide,
                )
    return x_commands, i_to_viz
