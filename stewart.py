import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import transform as tf


class StewartAxis:
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def get_transform(self):
        return tf.xyz_rpy_to_matrix((self.x, self.y, self.z, self.roll, self.pitch, self.yaw))

    def get_translation(self, dim=3):
        translation = np.ones((dim, 1))
        translation[0, 0] = self.x
        translation[1, 0] = self.y
        translation[2, 0] = self.z
        return translation

    def get_rotation(self):
        return tf.rpy_to_matrix((self.roll, self.pitch, self.yaw))


class StewartLink:
    def __init__(self, lower_joint: StewartAxis, upper_joint: StewartAxis, crank, rocker, initial_value=0.0):
        self.lower_joint = lower_joint
        self.upper_joint = upper_joint
        self.crank = crank
        self.rocker = rocker
        self.value = initial_value

    def get_middle_joint_at_lower_axis(self, dim=3):
        return (self.lower_joint.get_transform()
                @ tf.xyz_rpy_to_matrix((0.0, 0.0, 0.0, 0.0, 0.0, self.value))
                @ np.array([[self.rocker], [0.0], [0.0], [1.0]]))[:dim, [0]]


class Stewart:
    def __init__(self, lower_axis: StewartAxis, links: list[StewartLink], initial_upper_axis=StewartAxis()):
        self.lower_axis = lower_axis
        self.links = links
        self.upper_axis = initial_upper_axis
        self.solve()

    def set_target(self, upper_axis: StewartAxis):
        self.upper_axis = upper_axis
        self.solve()

    def get_thetas(self):
        thetas = []
        for link in self.links:
            thetas.append(link.value)
        return thetas

    def solve(self, answer=2):
        upper_to_lower_axis = tf.inverse_transform(self.lower_axis.get_transform()) @ self.upper_axis.get_transform()
        for link in self.links:
            # transform all point to lower joint axis
            upper_to_lower_joint = tf.inverse_transform(link.lower_joint.get_transform()) @ upper_to_lower_axis

            # calculation
            upper_joint = (upper_to_lower_joint @ link.upper_joint.get_translation(4))[:3, [0]]
            phi = np.arctan2(upper_joint[0], upper_joint[1])
            x = (np.sum(np.power(upper_joint, 2)) + np.power(link.rocker, 2) - np.power(link.crank, 2)) / (
                        2 * link.rocker * np.linalg.norm(upper_joint[:2, :]))

            # no solution
            if x > 1 or x < -1:
                raise RuntimeError('Stewart no solution.')

            # multiple solutions
            if answer == 1:
                theta = np.arcsin(x) - phi
            else:
                theta = math.pi - np.arcsin(x) - phi

            # theta -> [-pi, pi]
            if theta > math.pi:
                theta -= 2 * math.pi
            if theta < -math.pi:
                theta += 2 * math.pi
            link.value = theta

    def get_points(self):
        lower_to_world = self.lower_axis.get_transform()
        upper_to_world = self.upper_axis.get_transform()
        points = np.zeros((6, 3, 3))
        for i in range(6):
            link = self.links[i]
            points[i, 0, :] = (lower_to_world @ link.lower_joint.get_translation(4))[:3, 0]
            points[i, 1, :] = (lower_to_world @ link.get_middle_joint_at_lower_axis(4))[:3, 0]
            points[i, 2, :] = (upper_to_world @ link.upper_joint.get_translation(4))[:3, 0]

        return points

    def interact(self):
        fig = plt.figure()
        ax1 = fig.add_axes([0.0, 0.2, 0.8, 0.8], projection='3d')
        ax2 = fig.add_axes([0.1, 0.15, 0.8, 0.05])
        ax3 = fig.add_axes([0.1, 0.1, 0.8, 0.05])
        ax4 = fig.add_axes([0.1, 0.05, 0.8, 0.05])
        ax5 = fig.add_axes([0.95, 0.25, 0.05, 0.65])
        ax6 = fig.add_axes([0.9, 0.25, 0.05, 0.65])
        ax7 = fig.add_axes([0.85, 0.25, 0.05, 0.65])
        ax8 = fig.add_axes([0.8, 0.01, 0.1, 0.04])
        target_x_slider = Slider(
            ax=ax2,
            label='x',
            valmin=-0.2,
            valmax=0.2,
            valinit=0.0
        )
        target_y_slider = Slider(
            ax=ax3,
            label='y',
            valmin=-0.2,
            valmax=0.2,
            valinit=0.0
        )
        target_z_slider = Slider(
            ax=ax4,
            label='z',
            valmin=0.0,
            valmax=0.3,
            valinit=0.2
        )
        target_roll_slider = Slider(
            ax=ax5,
            label='roll',
            valmin=-3.0,
            valmax=3.0,
            valinit=0.0,
            orientation="vertical"
        )
        target_pitch_slider = Slider(
            ax=ax6,
            label='pitch',
            valmin=-3.0,
            valmax=3.0,
            valinit=0.0,
            orientation="vertical"
        )
        target_yaw_slider = Slider(
            ax=ax7,
            label='yaw',
            valmin=-3.0,
            valmax=3.0,
            valinit=0.0,
            orientation="vertical"
        )
        button = Button(ax8, 'Reset', hovercolor='0.975')

        points = self.get_points()
        points = np.append(points, points[[0], :, :], axis=0)

        lower_line = ax1.plot(points[:, 0, 0], points[:, 0, 1], points[:, 0, 2])[0]
        upper_line = ax1.plot(points[:, 2, 0], points[:, 2, 1], points[:, 2, 2])[0]

        link_lines = []
        for i in range(6):
            link_lines.append(ax1.plot(points[i, :, 0], points[i, :, 1], points[i, :, 2])[0])

        def update(attr, val):
            setattr(self.upper_axis, attr, val)
            try:
                self.solve()
            except RuntimeError:
                fig.suptitle('No solution')
            else:
                fig.suptitle('')
                new_points = self.get_points()
                new_points = np.append(new_points, new_points[[0], :, :], axis=0)
                lower_line.set_data_3d(new_points[:, 0, 0], new_points[:, 0, 1], new_points[:, 0, 2])
                upper_line.set_data_3d(new_points[:, 2, 0], new_points[:, 2, 1], new_points[:, 2, 2])
                for i in range(6):
                    link_lines[i].set_data_3d(new_points[i, :, 0], new_points[i, :, 1], new_points[i, :, 2])
                fig.canvas.draw_idle()

        def reset(event):
            target_x_slider.reset()
            target_y_slider.reset()
            target_z_slider.reset()
            target_roll_slider.reset()
            target_pitch_slider.reset()
            target_yaw_slider.reset()

        target_x_slider.on_changed(lambda val: update('x', val))
        target_y_slider.on_changed(lambda val: update('y', val))
        target_z_slider.on_changed(lambda val: update('z', val))
        target_roll_slider.on_changed(lambda val: update('roll', val))
        target_pitch_slider.on_changed(lambda val: update('pitch', val))
        target_yaw_slider.on_changed(lambda val: update('yaw', val))
        button.on_clicked(reset)

        ax1.set_aspect('equal')
        plt.show()

