import os
import pickle

import cmocean as cmo
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from stopro.data_generator.drag import Drag
from stopro.data_generator.stokes_2D_generator import StokesDataGenerator

# from stopro.data_handler.data_handle_module import HdfOperator


class Cylinder(Drag):
    """data generator for flow past a cylinder"""

    def __init__(
        self,
        L=1.0,
        H=1.0,
        particle_radius=0.1953125,  # 0.0390625
        particle_center=0.5,
        particle_y_velocity=0.0,
        slide=0.03,
        random_arrange=False,
        delta_p=-1.0,
        use_force_as_constant_pressure=True,
        use_u_inlet=False,
        use_difu=True,
        use_difp=True,
        use_gradp_training=False,
    ):
        super().__init__(
            L,
            H,
            particle_radius,
            particle_center,
            particle_y_velocity,
            slide,
            random_arrange,
        )
        self.delta_p = delta_p
        self.use_force_as_constant_pressure = use_force_as_constant_pressure
        self.use_u_inlet = use_u_inlet

    def change_outside_values_to_zero(self, r, f):
        is_inside_the_domain = self.get_index_in_domain(r)
        f_new = np.zeros(len(f))
        f_new[is_inside_the_domain] = f[is_inside_the_domain]
        return f_new

    def generate_training_data(
        self,
        u_num_surface=None,
        u_num_wall=None,
        f_num=None,
        f_pad=None,
        div_num=None,
        div_pad=None,
        difu_num=None,
        difp_num=None,
        num_inner=5,
        dr=0.2,
    ):
        self.r = []
        self.f = []
        self.generate_u(u_num_surface, u_num_wall)
        self.generate_difu(difu_num)
        self.generate_f(f_num, f_pad, num_inner, dr)
        self.generate_div(div_num, div_pad, num_inner, dr)
        self.generate_difp(difp_num)
        return self.r, self.f

    def generate_test(self, test_num=None, infer_governing_eqs=False):
        try:
            with open(
                # "/work/jh210017a/q24015/template_data/0314_cylinder_test_484.pickle",
                "/work/jh210017a/q24015/template_data/0501_cylinder_25_test_484.pickle",
                "rb",
            ) as file:
                save_dict = pickle.load(file)
        except:
            with open(
                "/home/ogawa_kenta/template_data/0308_drag_test_676.pickle", "rb"
            ) as file:
                save_dict = pickle.load(file)
        # except:
        #     raise FileNotFoundError('File was not found')
        r, ux, uy = save_dict["r"], save_dict["ux"], save_dict["uy"]

        # r = self.make_r_mesh(0., self.L, 0., self.H, test_num, test_num)
        # ux = np.ones(len(r))
        # uy = np.ones(len(r))

        self.r_test += [r, r]
        self.f_test += [ux, uy]
        return self.r_test, self.f_test

    def generate_u(self, u_num_surface, u_num_wall, use_inner=False):
        # dirichlet b.c. on the surface of the cylinder
        # use_inner = True
        r_ux_surface = self.make_r_surface(u_num_surface, use_inner)
        ux_surface = np.zeros(len(r_ux_surface))
        r_uy_surface = self.make_r_surface(u_num_surface, use_inner)
        uy_surface = np.full(len(r_uy_surface), self.particle_y_velocity)
        # non-slip dirichlet b.c. on the wall
        r_ux_wall = self.make_r_mesh(0.0, self.L, 0.0, self.H, u_num_wall, 2)
        r_uy_wall = self.make_r_mesh(0.0, self.L, 0.0, self.H, u_num_wall, 2)
        ux_wall = np.zeros(len(r_ux_wall))
        uy_wall = np.zeros(len(r_uy_wall))
        if self.use_u_inlet:
            with open(
                # "/work/jh210017a/q24015/template_data/0314_cylinder_test_484.pickle",
                "/work/jh210017a/q24015/template_data/0501_cylinder_25_test_484.pickle",
                "rb",
            ) as file:
                save_dict = pickle.load(file)
            r, ux, uy = save_dict["r"], save_dict["ux"], save_dict["uy"]
            index_inlet = r[:, 0] == 0.0
            # inletは全部で22点ある
            split = 6
            r_inlet = r[index_inlet][1:-1]
            ux_inlet = ux[index_inlet][1:-1]
            uy_inlet = uy[index_inlet][1:-1]
            # concatenate
            r_ux = np.concatenate([r_ux_surface, r_ux_wall, r_inlet])
            r_uy = np.concatenate([r_uy_surface, r_uy_wall, r_inlet])
            ux = np.concatenate([ux_surface, ux_wall, ux_inlet])
            uy = np.concatenate([uy_surface, uy_wall, uy_inlet])
        else:
            # concatenate
            r_ux = np.concatenate([r_ux_surface, r_ux_wall])
            r_uy = np.concatenate([r_uy_surface, r_uy_wall])
            ux = np.concatenate([ux_surface, ux_wall])
            uy = np.concatenate([uy_surface, uy_wall])

        self.r += [r_ux, r_uy]
        self.f += [ux, uy]

    def generate_f(self, f_num, f_pad, num_inner, dr):
        r_fx = self.make_r_mesh_mixed(num_inner, f_num, dr, f_pad)
        r_fy = self.make_r_mesh_mixed(num_inner, f_num, dr, f_pad)
        if self.use_force_as_constant_pressure:
            force_x = -self.delta_p / self.L
        else:
            force_x = 0.0
        force_y = 0.0
        fx = np.full(len(r_fx), force_x)
        fy = np.full(len(r_fy), force_y)

        self.r += [r_fx, r_fy]
        self.f += [fx, fy]

    def generate_div(self, div_num, div_pad, num_inner, dr, div=0.0):
        r_div = self.make_r_mesh_mixed(num_inner, div_num, dr, div_pad)
        div = np.full(len(r_div), div)

        self.r += [r_div]
        self.f += [div]

    def generate_difu(self, difu_num):
        r_difu = self.make_r_mesh(
            0, self.L, self.slide, self.H - self.slide, 1, difu_num
        )
        difu = np.zeros(len(r_difu))
        self.r += [r_difu, r_difu]
        self.f += [difu, difu]

    def generate_difp(self, difp_num):
        r_difp = self.make_r_mesh(
            0, self.L, self.slide, self.H - self.slide, 1, difp_num
        )
        if self.use_force_as_constant_pressure:
            difp = np.zeros(len(r_difp))
        else:
            difp = np.full(len(r_difp), self.delta_p)
        self.r += [r_difp]
        self.f += [difp]

    def plot_train(self, save=False, path=None, show=False):
        num_surface = 100
        r_surface = self.make_r_surface(num_surface)

        ms = 2
        ms2 = 2
        fig, axs = plt.subplots(
            figsize=(3 * 3, 3 * 2), nrows=2, ncols=3, sharex=True, sharey=True
        )
        clrs = [
            self.COLOR["dark_highlight"],
            self.COLOR["dark_highlight"],
            self.COLOR["superfine"],
            self.COLOR["superfine"],
            "darkblue",
            "darkblue",
            "darkblue",
            self.COLOR["superfine"],
            self.COLOR["light_highlight"],
        ]
        lbls = ["ux", "uy", "p", "fx", "fy", "div"]
        axes = axs.reshape(-1)
        for i, ax in enumerate(axes):
            if i == 0 or i == 1:
                ax.set_title(lbls[i])
                ax.plot(
                    self.r[i][:, 0],
                    self.r[i][:, 1],
                    ls="None",
                    marker="o",
                    color=clrs[i],
                    ms=ms2,
                    label="b.c.",
                )
                index = i + 2
                ax.plot(
                    self.r[index][:, 0],
                    self.r[index][:, 1],
                    ls="None",
                    marker="o",
                    color=self.COLOR["superfine"],
                    ms=ms,
                    label="$x$ p.b.c.",
                )
                # if i == 1:
                #     ax.legend(loc="upper right",
                #               bbox_to_anchor=(0., 1.02,), fontsize='small')
            elif i == 2:
                ax.set_title(lbls[i])
                index = 7
                ax.plot(
                    self.r[index][:, 0],
                    self.r[index][:, 1],
                    ls="None",
                    marker="o",
                    color=clrs[index],
                    ms=ms,
                    label="$x$ p.b.c.",
                )
            else:
                index = i + 1
                ax.set_title(lbls[i])
                ax.plot(
                    self.r[index][:, 0],
                    self.r[index][:, 1],
                    ls="None",
                    marker="o",
                    color=clrs[index],
                    ms=ms2,
                )
            ax.set_aspect("equal", adjustable="box")
            ax.plot(r_surface[:, 0], r_surface[:, 1], color="k")
        plt.tight_layout()
        if save:
            dir_path = f"{path}/fig"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            fig.savefig(f"{dir_path}/train.png")
        if show:
            plt.show()
        plt.clf()
        plt.close()

    def plot_test(self, save=False, path=None, val_limits=None):
        fig, axs = plt.subplots(figsize=(5 * 2, 5), ncols=2, sharey=True)
        r_ux = self.r_test[0]
        num_per_side = int(np.sqrt(len(r_ux)))
        x_grid = np.linspace(0.0, np.max(r_ux[:, 0]), num_per_side)
        y_grid = np.linspace(0.0, np.max(r_ux[:, 1]), num_per_side)
        cmaps = [cmo.cm.dense, cmo.cm.balance]
        for i, ax in enumerate(axs):
            f_mesh = self.f_test[i].reshape(num_per_side, num_per_side)
            self.plot_heatmap(
                fig, ax, x_grid, y_grid, f_mesh, shading="gouraud", cmap=cmaps[i]
            )
        fig.tight_layout()
        if save:
            dir_path = f"{path}/fig"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            fig.savefig(f"{dir_path}/test.png")
        plt.clf()
        plt.close()
