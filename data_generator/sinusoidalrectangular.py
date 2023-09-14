import os
import pickle

import cmocean as cmo
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, vmap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy import integrate

from stopro.data_generator.sinusoidalcylinder import SinusoidalCylinder
from stopro.data_handler.data_handle_module import HdfOperator


class SinusoidalRectangular(SinusoidalCylinder):
    def make_r_surface(self, num, u_num_inner=0):
        # if not use_inner:
        #     r_mesh, θ_mesh = np.meshgrid(self.particle_radius, θ)
        # else:
        #     radius_range = np.linspace(0.013, self.particle_radius, 3)
        #     r_mesh, θ_mesh = np.meshgrid(radius_range, θ)
        num_per_side = int(num / 4)
        x_start = self.particle_center[0] - self.particle_radius
        x_end = self.particle_center[0] + self.particle_radius
        y_start = self.particle_center[1] - self.particle_radius
        y_end = self.particle_center[1] + self.particle_radius
        r_left_right = self.make_r_mesh(x_start, x_end, y_start, y_end, 2, num_per_side)
        r_top_bottom = self.make_r_mesh(x_start, x_end, y_start, y_end, num_per_side, 2)
        index_delete = (r_top_bottom[:, 0] == x_start) | (r_top_bottom[:, 0] == x_end)
        r_top_bottom = r_top_bottom[~index_delete]
        r = np.concatenate([r_left_right, r_top_bottom])
        return r

    def get_index_out_cylinder(self, r, radius_min=None):
        rx = np.abs(r[:, 0] - self.particle_center[0])
        ry = np.abs(r[:, 1] - self.particle_center[1])
        if not radius_min:
            # index_out_cylinder = np.where(
            #     rx**2 + ry**2 > self.particle_radius**2
            # )[0]
            index_out_cylinder = np.where(
                (rx > self.particle_radius) | (ry > self.particle_radius)
            )[0]
        else:
            index_out_cylinder = np.where((rx > radius_min) | (ry > radius_min))[0]
        return index_out_cylinder
