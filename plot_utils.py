"""Matplotlib artists."""
import numpy as np
from matplotlib import artist
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3D, Text3D


class Frame(artist.Artist):
    """A Matplotlib artist that displays a frame represented by its basis.
    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B
    label : str, optional (default: None)
        Name of the frame
    s : float, optional (default: 1)
        Length of basis vectors
    draw_label_indicator : bool, optional (default: True)
        Controls whether the line from the frame origin to frame label is
        drawn.
    Other arguments except 'c' and 'color' are passed on to Line3D.
    """
    def __init__(self, A2B, label=None, s=1.0, **kwargs):
        super(Frame, self).__init__()

        if "c" in kwargs:
            kwargs.pop("c")
        if "color" in kwargs:
            kwargs.pop("color")

        self.draw_label_indicator = kwargs.pop("draw_label_indicator", True)

        self.s = s

        self.x_axis = Line3D([], [], [], color="r", **kwargs)
        self.y_axis = Line3D([], [], [], color="g", **kwargs)
        self.z_axis = Line3D([], [], [], color="b", **kwargs)

        self.draw_label = label is not None
        self.label = label

        if self.draw_label:
            if self.draw_label_indicator:
                self.label_indicator = Line3D([], [], [], color="k", **kwargs)
            self.label_text = Text3D(0, 0, 0, text="", zdir="x")

        self.set_data(A2B, label)

    def set_data(self, A2B, label=None):
        """Set the transformation data.
        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Transform from frame A to frame B
        label : str, optional (default: None)
            Name of the frame
        """
        R = A2B[:3, :3]
        p = A2B[:3, 3]

        for d, b in enumerate([self.x_axis, self.y_axis, self.z_axis]):
            b.set_data(np.array([p[0], p[0] + self.s * R[0, d]]),
                       np.array([p[1], p[1] + self.s * R[1, d]]))
            b.set_3d_properties(np.array([p[2], p[2] + self.s * R[2, d]]))

        if self.draw_label:
            if label is None:
                label = self.label
            label_pos = p + 0.5 * self.s * (R[:, 0] + R[:, 1] + R[:, 2])

            if self.draw_label_indicator:
                self.label_indicator.set_data(
                    np.array([p[0], label_pos[0]]),
                    np.array([p[1], label_pos[1]]))
                self.label_indicator.set_3d_properties(
                    np.array([p[2], label_pos[2]]))

            self.label_text.set_text(label)
            self.label_text.set_position([label_pos[0], label_pos[1]])
            self.label_text.set_3d_properties(label_pos[2], zdir="x")

    @artist.allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        """Draw the artist."""
        for b in [self.x_axis, self.y_axis, self.z_axis]:
            b.draw(renderer, *args, **kwargs)
        if self.draw_label:
            if self.draw_label_indicator:
                self.label_indicator.draw(renderer, *args, **kwargs)
            self.label_text.draw(renderer, *args, **kwargs)
        super(Frame, self).draw(renderer, *args, **kwargs)

    def add_frame(self, axis):
        """Add the frame to a 3D axis."""
        for b in [self.x_axis, self.y_axis, self.z_axis]:
            axis.add_line(b)
        if self.draw_label:
            if self.draw_label_indicator:
                axis.add_line(self.label_indicator)
            axis._add_text(self.label_text)
