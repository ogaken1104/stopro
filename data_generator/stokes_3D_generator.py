import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from stopro.data_generator.data_generator import DataGenerator


class Stokes3DGenerator(DataGenerator):
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
