import math
from stewart import Stewart, StewartAxis, StewartLink


P1 = 0.1
P2 = 0.1
P3 = 0.2
P4 = 0.2
P5 = 0.05
P6 = 0.2
P7 = 0.1
P8 = 0.03
P9 = 0.02


def create_stewart():
    lower_axis = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    lower_joints = ((P1 * math.cos(math.pi / 3) + P8 * math.sin(math.pi / 3), P1 * math.sin(math.pi / 3) - P8 * math.cos(math.pi / 3), 0.0, -math.pi / 2, 0.0, -math.pi / 6),
                    (P1 * math.cos(math.pi / 3) - P8 * math.sin(math.pi / 3), P1 * math.sin(math.pi / 3) + P8 * math.cos(math.pi / 3), 0.0, -math.pi / 2, 0.0, 5 * math.pi / 6),
                    (-P1, P8, 0.0, -math.pi / 2, 0.0, math.pi / 2),
                    (-P1, -P8, 0.0, -math.pi / 2, 0.0, -math.pi / 2),
                    (P1 * math.cos(math.pi / 3) - P8 * math.sin(math.pi / 3), -P1 * math.sin(math.pi / 3) - P8 * math.cos(math.pi / 3), 0.0, -math.pi / 2, 0.0, -5 * math.pi / 6),
                    (P1 * math.cos(math.pi / 3) + P8 * math.sin(math.pi / 3), -P1 * math.sin(math.pi / 3) + P8 * math.cos(math.pi / 3), 0.0, -math.pi / 2, 0.0, math.pi / 6))
    upper_axis = (0.0, 0.0, P4, 0.0, 0.0, 0.0)
    upper_joints = ((P5 * math.cos(math.pi / 3) + P9 * math.sin(math.pi / 3), P5 * math.sin(math.pi / 3) - P9 * math.cos(math.pi / 3), 0.0, 0.0, 0.0, 0.0),
                    (P5 * math.cos(math.pi / 3) - P9 * math.sin(math.pi / 3), P5 * math.sin(math.pi / 3) + P9 * math.cos(math.pi / 3), 0.0, 0.0, 0.0, 0.0),
                    (-P5, P9, 0.0, 0.0, 0.0, 0.0),
                    (-P5, -P9, 0.0, 0.0, 0.0, 0.0),
                    (P5 * math.cos(math.pi / 3) - P9 * math.sin(math.pi / 3), -P5 * math.sin(math.pi / 3) - P9 * math.cos(math.pi / 3), 0.0, 0.0, 0.0, 0.0),
                    (P5 * math.cos(math.pi / 3) + P9 * math.sin(math.pi / 3), -P5 * math.sin(math.pi / 3) + P9 * math.cos(math.pi / 3), 0.0, 0.0, 0.0, 0.0))
    crank = P6
    rocker = P7

    links = []
    for lower_joint, upper_joint in zip(lower_joints, upper_joints):
        lower_joint = StewartAxis(*lower_joint)
        upper_joint = StewartAxis(*upper_joint)
        links.append(StewartLink(lower_joint, upper_joint, crank, rocker))

    return Stewart(StewartAxis(*lower_axis), links, StewartAxis(*upper_axis))


if __name__ == '__main__':
    stewart = create_stewart()
    stewart.interact()

