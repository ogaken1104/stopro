from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import cmocean as cmo


from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from stopro.data_generator.stokes_3D_generator import Stokes3DGenerator


class Drag3D(Stokes3DGenerator):
    def __init__(
        self,
        particle_radius: float = 0.1,
        L: float = 1.0,
        constant_velocity: float = 1.0,
        seed: int = 0,
    ):
        self.a = particle_radius
        self.L = L
        self.U0 = constant_velocity
        self.r = []
        self.f = []
        self.r_test = []
        self.f_test = []
        self.seed = seed

    def func_u(self, r):
        x = r[:, 0]
        y = r[:, 1]
        z = r[:, 2]
        r2 = np.sum(r**2, axis=1)
        r_norm = np.sqrt(r2)
        r3 = r2 * r_norm
        a2_r2 = self.a**2 / r2
        ux = self.U0 * (
            1
            - self.a / (4 * r_norm) * (3 + a2_r2)
            - 3 / 4 * self.a * x**2 / r3 * (1 - a2_r2)
        )
        uy = -self.U0 * 3 / 4 * self.a * x * y / r3 * (1 - a2_r2)
        uz = -self.U0 * 3 / 4 * self.a * x * z / r3 * (1 - a2_r2)
        ux = self.change_outside_values_to_zero(r, ux)
        uy = self.change_outside_values_to_zero(r, uy)
        uz = self.change_outside_values_to_zero(r, uz)
        return ux, uy, uz

    def get_index_in_domain(self, r, radius_min=None):
        rx = r[:, 0]
        ry = r[:, 1]
        rz = r[:, 2]
        if not radius_min:
            index_in_domain = np.where(rx**2 + ry**2 + rz**2 > self.a**2)[0]
        else:
            index_in_domain = np.where(rx**2 + ry**2 + rz**2 > radius_min**2)[0]
        return index_in_domain

    def make_r_mesh_sphere(self, num_per_side: int):
        r = self.make_r_mesh(
            -self.L,
            self.L,
            -self.L,
            self.L,
            -self.L,
            self.L,
            num_per_side,
            num_per_side,
            num_per_side,
        )
        r = self.delete_out_domain(r)
        return r

    def make_r_surface(self, num, z_plane=0.0):
        pass

    def generate_u(self, u_num, u_surface_num, slice_axis, slice_num):
        if slice_axis:
            x_num, y_num, z_num = u_num, u_num, u_num
            if slice_axis == "x":
                x_num = slice_num
            elif slice_axis == "y":
                y_num = slice_num
            elif slice_axis == "z":
                z_num = slice_num
            r = self.make_r_mesh(
                -self.L,
                self.L,
                -self.L,
                self.L,
                -self.L,
                self.L,
                x_num,
                y_num,
                z_num,
            )
            r = self.delete_out_domain(r)
        else:
            r = self.make_r_mesh_sphere(u_num)
        ux, uy, uz = self.func_u(r)
        if u_surface_num:
            r_surface = self.make_r_sphere_surface(u_surface_num, self.a)
            u_surface = np.zeros(len(r_surface))
            r = np.concatenate([r, r_surface])
            ux = np.concatenate([ux, u_surface])
            uy = np.concatenate([uy, u_surface])
            uz = np.concatenate([uz, u_surface])
        self.r += [r, r, r]
        self.f += [ux, uy, uz]

    def generate_f(self, f_num):
        r = self.make_r_mesh_sphere(f_num)
        f = np.zeros(len(r))
        self.r += [r, r, r]
        self.f += [f, f, f]

    def generate_div(self, div_num):
        r = self.make_r_mesh_sphere(div_num)
        div = np.zeros(len(r))
        self.r += [r]
        self.f += [div]

    def generate_test(self, test_num, z_plane=0.0):
        r = self.make_r_mesh(
            -self.L, self.L, -self.L, self.L, z_plane, z_plane, test_num, test_num, 1
        )
        ux, uy, uz = self.func_u(r)
        self.r_test += [r] * 3
        self.f_test += [ux, uy, uz]
        return self.r_test, self.f_test

    def generate_training_data(
        self,
        u_num,
        f_num,
        div_num,
        u_surface_num: int = 0,
        sigma2_noise: float = None,
        slice_axis: str = None,
        slice_num: int = None,
    ):
        self.generate_u(u_num, u_surface_num, slice_axis, slice_num)
        self.generate_f(f_num)
        self.generate_div(div_num)
        return self.r, self.f

    def plot_train(self, save=False, path=None, show=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")  # 3Dサブプロットを追加

        ax.scatter(self.r[0][:, 0], self.r[0][:, 1], self.r[0][:, 2], c="k")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.legend()
        plt.tight_layout()
        if save:
            dir_path = f"{path}/fig"
            fig.savefig(
                f"{dir_path}/train.png",
                bbox_inches="tight",
            )
        if show:
            plt.show()
        plt.clf()
        plt.close()

    def plot_test(self, save=False, path=None, show=False, val_limits=None):
        # num_surface = 100
        # r_surface = self.make_r_surface(num_surface)

        fig, axs = plt.subplots(figsize=(5 * 3, 3), ncols=3, sharex=True, sharey=True)
        y_num = int(np.sqrt(len(self.r_test[0])))
        x_num = y_num
        y_grid = np.linspace(-self.L, self.L, y_num)
        x_grid = np.linspace(-self.L, self.L, x_num)
        cmaps = [cmo.cm.dense, cmo.cm.balance, cmo.cm.balance]
        for i, ax in enumerate(axs):
            f_mesh = self.f_test[i].reshape(y_num, x_num)
            mappable = ax.pcolormesh(
                x_grid,
                y_grid,
                f_mesh,
                cmap=cmaps[i],
                vmin=val_limits[i][0],
                vmax=val_limits[i][1],
                shading="nearest",
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(mappable, cax=cax)
            ax.set_aspect("equal", adjustable="box")
            # ax.plot(r_surface[:, 0], r_surface[:, 1], color="k")
        # ax.legend()
        if save:
            dir_path = f"{path}/fig"
            fig.savefig(
                f"{dir_path}/test.png",
                bbox_inches="tight",
            )
        if show:
            plt.show()
        plt.clf()
        plt.close()
