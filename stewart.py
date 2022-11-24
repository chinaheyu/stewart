import numpy as np
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

    def get_middle_joint_translation(self, dim=3):
        return (self.get_lower_joint_transform() @ np.array([[self.rocker], [0.0], [0.0], [1.0]]))[:dim, [0]]

    def get_lower_joint_transform(self):
        return self.lower_joint.get_transform() @ tf.xyz_rpy_to_matrix((0.0, 0.0, 0.0, 0.0, 0.0, self.value))


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
                theta = np.pi - np.arcsin(x) - phi

            # theta -> [-pi, pi]
            if theta > np.pi:
                theta -= 2 * np.pi
            if theta < -np.pi:
                theta += 2 * np.pi
            link.value = theta

        # check error
        assert self.max_error() < 1e-3

    def max_error(self):
        points = self.get_points()
        solved = np.linalg.norm((points[:, 1, :] - points[:, 2, :]), axis=1)
        ground_truth = np.array([i.crank for i in self.links])
        error = (solved - ground_truth) / ground_truth
        return np.max(np.abs(error))

    def get_points(self):
        lower_to_world = self.lower_axis.get_transform()
        upper_to_world = self.upper_axis.get_transform()
        points = np.zeros((6, 3, 3))
        for i in range(6):
            link = self.links[i]
            points[i, 0, :] = (lower_to_world @ link.lower_joint.get_translation(4))[:3, 0]
            points[i, 1, :] = (lower_to_world @ link.get_middle_joint_translation(4))[:3, 0]
            points[i, 2, :] = (upper_to_world @ link.upper_joint.get_translation(4))[:3, 0]

        return points

    def get_lower_joint_transforms(self):
        lower_to_world = self.lower_axis.get_transform()
        lower_joint_transforms = []
        for link in self.links:
            lower_joint_transforms.append(lower_to_world @ link.get_lower_joint_transform())
        return lower_joint_transforms
