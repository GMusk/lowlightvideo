import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(trajectory, smoothed_trajectory):
    """Plot video trajectory
    Create a plot of the video's trajectory & smoothed trajectory.
    Separate subplots are used to show the x and y trajectory.
    :param transforms: VidStab transforms attribute
    :param trajectory: VidStab trajectory attribute
    :param smoothed_trajectory: VidStab smoothed_trajectory attribute
    :return: tuple of matplotlib objects ``(Figure, (AxesSubplot, AxesSubplot))``
    """

    with plt.style.context('ggplot'):
        fig, (ax1, ax2) = plt.subplots(2, sharex='all')

        # x trajectory
        ax1.plot(trajectory[:, 0], label='Trajectory')
        ax1.plot(smoothed_trajectory[:, 0], label='Smoothed Trajectory')
        ax1.set_ylabel('dx')

        # y trajectory
        ax2.plot(trajectory[:, 1], label='Trajectory')
        ax2.plot(smoothed_trajectory[:, 1], label='Smoothed Trajectory')
        ax2.set_ylabel('dy')

        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')

        plt.xlabel('Frame Number')

        fig.suptitle('Video Trajectory', x=0.15, y=0.96, ha='left')
        fig.canvas.set_window_title('Trajectory')

        plt.savefig("tragectory.png")

        return fig, (ax1, ax2)
