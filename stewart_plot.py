from PySide6.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QSpacerItem, \
    QSizePolicy, QCheckBox
from PySide6.QtCore import Qt, Slot, QTimer
from functools import partial
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from stewart import *
import trimesh
import transform as tf
import time
import copy
import pathlib

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def hex_to_rgba(h):
    '''Takes a hex rgb string (e.g. #ffffff) and returns an RGB tuple (float, float, float).'''
    return *tuple(int(h[i:i + 2], 16) / 255. for i in (1, 3, 5)), 1.0  # skip '#'


class StewartPlot(QWidget):
    def __init__(self, stewart: Stewart):
        super().__init__()
        self.stewart = stewart
        self.origin_upper_axis = copy.copy(self.stewart.upper_axis)

        self.layout = QVBoxLayout(self)

        self.view = gl.GLViewWidget()
        self.layout.addWidget(self.view)

        scale = np.max([i.crank for i in self.stewart.links]) + np.max([i.rocker for i in self.stewart.links])
        self.sliders = {
            'x': (self.stewart.lower_axis.x - scale, self.stewart.lower_axis.x + scale, self.stewart.lower_axis.x),
            'y': (self.stewart.lower_axis.y - scale, self.stewart.lower_axis.y + scale, self.stewart.lower_axis.y),
            'z': (self.stewart.lower_axis.z, self.stewart.lower_axis.z + scale, self.stewart.upper_axis.z),
            'roll': (self.stewart.lower_axis.roll - np.pi / 2, self.stewart.lower_axis.roll + np.pi / 2,
                     self.stewart.lower_axis.roll),
            'pitch': (self.stewart.lower_axis.pitch - np.pi / 2, self.stewart.lower_axis.pitch + np.pi / 2,
                      self.stewart.lower_axis.pitch),
            'yaw': (self.stewart.lower_axis.yaw - np.pi / 2, self.stewart.lower_axis.yaw + np.pi / 2,
                    self.stewart.lower_axis.yaw)}

        self.sliders_container = dict()

        # create slider
        layout = QGridLayout()
        for i, name in enumerate(self.sliders.keys()):
            label = QLabel()
            label.setText(name)
            layout.addWidget(label, i, 0)

            slider = QSlider()
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setValue(self.value_to_slider(name, self.sliders[name][2]))
            slider.setOrientation(Qt.Orientation.Horizontal)
            layout.addWidget(slider, i, 1)
            self.sliders_container[name] = slider

            label = QLabel()
            label.setText(f'{self.sliders[name][2]:0.3f}')
            layout.addWidget(label, i, 3)

            slider.valueChanged.connect(partial(self.slider_changed, label, name))

            self.layout.addLayout(layout)

        layout = QHBoxLayout()
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum))
        button = QPushButton()
        button.setText('Reset')
        button.clicked.connect(self.reset)
        layout.addWidget(button)
        check_box = QCheckBox()
        check_box.setText('Animation')
        check_box.stateChanged.connect(self.animation)
        layout.addWidget(check_box)
        self.layout.addLayout(layout)

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animation_function)
        self.animation_timer.setInterval(16)

        self.z_grid = gl.GLGridItem()
        self.view.addItem(self.z_grid)

        self.lower_platform = gl.GLLinePlotItem()
        self.view.addItem(self.lower_platform)

        self.upper_platform = gl.GLLinePlotItem()
        self.view.addItem(self.upper_platform)

        self.links = [gl.GLLinePlotItem() for _ in range(6)]
        for link in self.links:
            self.view.addItem(link)

        mesh = trimesh.load_mesh(pathlib.Path(__file__).resolve().with_name('head.STL'))

        vertices = mesh.vertices * 20.0
        vertices = np.hstack((vertices, np.ones((len(vertices), 1)))).T
        vertices = tf.xyz_rpy_to_matrix((0, 0, 0, 0, 0, -np.pi / 2)) @ vertices
        mesh.vertices = vertices[:3, :].T
        color = [hex_to_rgba('#F1C27D')]
        self.mesh = gl.GLMeshItem(vertexes=mesh.vertices,
                                  faces=mesh.faces,
                                  faceColors=np.array(color * mesh.faces.shape[0]),
                                  vertexColors=np.array(color * mesh.vertices.shape[0]),
                                  shader='shaded')
        self.view.addItem(self.mesh)
        self.view.setCameraParams(
            **{'elevation': 31.0, 'center': pg.Vector(0.967638, 0.181012, 0.850586), 'azimuth': 9.0,
               'distance': 0.05726895335904424, 'fov': 60})

        self.plot()

    @Slot()
    def animation(self, value):
        if value == Qt.CheckState.Checked.value:
            self.animation_timer.start()
        else:
            self.animation_timer.stop()

    @Slot()
    def animation_function(self):
        # self.stewart.upper_axis.roll = 0.2 * np.sin(2 * np.pi / 1.0 * time.time())
        # self.stewart.upper_axis.pitch = 0.2 * np.cos(2 * np.pi / 1.0 * time.time())
        # self.stewart.upper_axis.x = self.origin_upper_axis.x + 0.05 * np.sin(2 * np.pi / 1.0 * time.time())
        # self.stewart.upper_axis.y = self.origin_upper_axis.y + 0.05 * np.cos(2 * np.pi / 1.0 * time.time())
        # self.stewart.solve()
        # self.plot()
        self.sliders_container['roll'].setValue(
            self.value_to_slider('roll', 0.2 * np.sin(2 * np.pi / 1.0 * time.time())))
        self.sliders_container['pitch'].setValue(
            self.value_to_slider('pitch', 0.2 * np.cos(2 * np.pi / 1.0 * time.time())))

    @Slot()
    def reset(self):
        for name in self.sliders.keys():
            self.sliders_container[name].setValue(self.value_to_slider(name, self.sliders[name][2]))

    def slider_changed(self, label, name, value):
        origin = getattr(self.stewart.upper_axis, name)
        setattr(self.stewart.upper_axis, name, self.slider_to_value(name, value))
        try:
            self.stewart.solve()
        except RuntimeError:
            setattr(self.stewart.upper_axis, name, origin)
        else:
            label.setText(f'{self.slider_to_value(name, value):0.3f}')
            self.plot()

    def slider_to_value(self, name, value):
        return (self.sliders[name][1] - self.sliders[name][0]) * value / 1000. + self.sliders[name][0]

    def value_to_slider(self, name, value):
        return 1000. * (value - self.sliders[name][0]) / (self.sliders[name][1] - self.sliders[name][0])

    def plot(self):
        points = self.stewart.get_points()
        points = np.append(points, points[[0], :, :], axis=0)

        self.lower_platform.setData(pos=points[:, 0, :], color=hex_to_rgba(colors[0]), width=5)
        self.upper_platform.setData(pos=points[:, 2, :], color=hex_to_rgba(colors[1]), width=5)

        for i in range(6):
            self.links[i].setData(pos=points[i, :, :], color=hex_to_rgba(colors[i + 2]), width=2)

        self.mesh.setTransform(self.stewart.upper_axis.get_transform())
