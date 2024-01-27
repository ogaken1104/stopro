import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from stopro.data_generator.data_generator import DataGenerator
from stopro.data_generator.stokes_2D_generator import StokesDataGenerator


class Stokes3DGenerator(StokesDataGenerator):
    def __init__(self):
        pass

    def make_r_mesh(
        self, x_start, x_end, y_start, y_end, z_start, z_end, numx, numy, numz
    ):
        x = np.linspace(x_start, x_end, numx)
        y = np.linspace(y_start, y_end, numy)
        z = np.linspace(z_start, z_end, numz)
        xx, yy, zz = np.meshgrid(x, y, z)
        r = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=1)
        return r

    @staticmethod
    def make_r_sphere_surface(num_points, radius):
        """
        球の表面上に等間隔で点を生成する関数

        Args:
            num_points(int): 生成する点の数
            radius(float): 球の半径

        Returns:
            points(np.ndarray): 生成された点の座標 (3次元 numpy 配列)
        """
        x = np.empty(0)
        y = np.empty(0)
        z = np.empty(0)
        phi = np.linspace(0.0001, np.pi - 0.0001, num_points)
        for ph in phi:
            num_theta = int(np.sin(ph) * num_points * 2)
            theta = np.linspace(0.0, 2.0 * np.pi, num_theta, endpoint=False)
            tt, pp = np.meshgrid(theta, ph)
            _x = np.sin(pp) * np.cos(tt) * radius
            _y = np.sin(pp) * np.sin(tt) * radius
            _z = np.cos(pp) * radius
            x = np.append(x, _x)
            y = np.append(y, _y)
            z = np.append(z, _z)
        points = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)
        return points
