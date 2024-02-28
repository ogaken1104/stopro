import os
import pickle

import cmocean as cmo
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, vmap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy import integrate
from pathlib import Path

from stopro.data_generator.cylinder import Cylinder
from stopro.data_generator.sinusoidal import Sinusoidal
from stopro.data_generator.stokes_2D_generator import StokesDataGenerator
from stopro.data_handler.data_handle_module import HdfOperator


class SinusoidalCylinder(Sinusoidal):
    def __init__(self, **kwargs):
        self.particle_radius = kwargs["particle_radius"]
        self.particle_center = np.array(kwargs["particle_center"])
        del kwargs["particle_radius"]
        del kwargs["particle_center"]
        ## need to modify
        super().__init__(**kwargs)

    def make_r_surface(self, num, u_num_inner=0):
        θ_start = 0
        θ_end = 2 * np.pi
        θ = np.linspace(θ_start, θ_end, num)
        r_mesh, θ_mesh = np.meshgrid(self.particle_radius, θ)
        # if not use_inner:
        #     r_mesh, θ_mesh = np.meshgrid(self.particle_radius, θ)
        # else:
        #     radius_range = np.linspace(0.013, self.particle_radius, 3)
        #     r_mesh, θ_mesh = np.meshgrid(radius_range, θ)
        if not self.random_arrange:
            pass
        else:
            np.random.seed(42)
            r_mesh -= (np.random.random_sample(r_mesh.shape)) * 0.01
            np.random.seed(42)
            θ_mesh += (np.random.random_sample(θ_mesh.shape) - 0.5) * 0.2
        xx, yy = r_mesh * np.cos(θ_mesh), r_mesh * np.sin(θ_mesh)
        r = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
        r += self.particle_center
        if u_num_inner:
            inner_pad = 0.05
            radius_generate = self.particle_radius - inner_pad
            r_inner = self.make_r_mesh(
                -radius_generate,
                radius_generate,
                -radius_generate,
                radius_generate,
                u_num_inner,
                u_num_inner,
            )
            r_inner += self.particle_center
            index_out_cylinder = self.get_index_out_cylinder(r_inner)
            r_inner = np.delete(r_inner, index_out_cylinder, axis=0)
            print(index_out_cylinder, r_inner)
            r = np.concatenate([r, r_inner])
        return r

    def get_index_out_cylinder(self, r, radius_min=None):
        rx = r[:, 0] - self.particle_center[0]
        ry = r[:, 1] - self.particle_center[1]
        if not radius_min:
            # index_out_cylinder = np.where(
            #     rx**2 + ry**2 > self.particle_radius**2
            # )[0]
            index_out_cylinder = np.where(rx**2 + ry**2 > self.particle_radius**2)[0]
        else:
            index_out_cylinder = np.where(rx**2 + ry**2 > radius_min**2)[0]
        return index_out_cylinder

    def get_index_in_channel(self, r):
        top_ok = r[:, 1] < self.calc_y_top(r[:, 0])
        bottom_ok = r[:, 1] > self.calc_y_bottom(r[:, 0])
        index_in_channel = top_ok & bottom_ok
        return index_in_channel

        # generation of training data

    def generate_u(
        self,
        u_num_surface,
        u_num_wall,
        u_num_inner=0,
        u_b=0.0,
        u_num_random=None,
    ):
        """
        premise: ux,uy values are taken at same points
        """
        # split u_num
        u_num_x, u_num_y = self.num_to_num_x_y(u_num_wall)
        u_pad = self.slide

        if self.use_random_u or self.use_random_and_boundary_u:
            # ########## when using spm as reference solution #################
            # # hdf_operatorでhdfに保存した計算結果を読み込み
            # # 点をランダムで抽出して使う
            # hdf_operator = HdfOperator(
            #     '/work/jh210017a/q24015/template_data/test_sinusoidal_from_spm_adimentionalized')
            # r_train, f_train = hdf_operator.load_train_data(
            #     ['r', 'f'], ['ux', 'uy'])
            # index_for_train = np.arange(0, len(r_train[0]), 1)
            # rng = np.random.default_rng(seed=43)
            # rng.shuffle(index_for_train)
            # index_for_train = index_for_train[:u_num]
            # r_u = r_train[0][index_for_train]
            # r_ux = r_u
            # r_uy = r_u
            # ux, uy = f_train[0][index_for_train], f_train[1][index_for_train]

            ########## when using fem as reference solution #################
            if self.__class__.__name__ == "SinusoidalCylinder":
                fname = "test_sinusoidalcylinder_fem/fem_train_2962.pickle"
            elif self.__class__.__name__ == "SinusoidalRectangular":
                fname = "test_sinusoidalrectangular_fem/fem_train_2930.pickle"
            with open(
                # f'{Path(os.path.abspath(__file__)).parent.parent}/template_data/test_sinusoidalcylinder_spm/0801_sinusoidalcylinder_train_24968.pickle',
                # f'{Path(os.path.abspath(__file__)).parent.parent}/template_data/test_sinusoidalcylinder_spm/0817_sinusoidalcylinder_train_2802.pickle',
                # f'{Path(os.path.abspath(__file__)).parent.parent}/template_data/test_sinusoidalcylinder_fem/fem_train_1524.pickle',
                f"{Path(os.path.abspath(__file__)).parent.parent}/template_data/{fname}",
                "rb",
            ) as file:
                save_dict = pickle.load(file)
            r_train = save_dict["r"]
            ux_train = save_dict["ux"]
            uy_train = save_dict["uy"]

            index_for_train = np.arange(0, len(r_train), 1)
            rng = np.random.default_rng(seed=self.seed)
            rng.shuffle(index_for_train)
            index_for_train = index_for_train[:u_num_random]
            if self.use_random_and_boundary_u:
                r_u_wall, u_wall = self.generate_wall_points(u_num_x)
                r_u_surface = self.make_r_surface(
                    u_num_surface, u_num_inner=u_num_inner
                )
                u_surface = np.zeros(len(r_u_surface))
                if self.use_inlet_u:
                    index_inlet = r_train[:, 0] == 0.0
                    index_for_random = np.intersect1d(
                        index_for_train,
                        np.arange(0, len(index_inlet), dtype=int)[~index_inlet],
                    )
                    # index_for_random = np.ones(len(index_for_random))[index_for_random]
                    r_u_random_all = r_train[index_for_random]
                    r_u_random = r_u_random_all
                    ux_random, uy_random = (
                        ux_train[index_for_random],
                        uy_train[index_for_random],
                    )
                    split = 2
                    r_u_inlet = r_train[index_inlet][::split]
                    ux_inlet = ux_train[index_inlet][::split]
                    uy_inlet = uy_train[index_inlet][::split]
                    ux = np.concatenate([u_wall, u_surface, ux_random, ux_inlet])
                    uy = np.concatenate([u_wall, u_surface, uy_random, uy_inlet])
                    r_u = np.concatenate([r_u_wall, r_u_surface, r_u_random, r_u_inlet])
                else:
                    r_u_random = r_train[index_for_train]
                    ux_random, uy_random = (
                        ux_train[index_for_train],
                        uy_train[index_for_train],
                    )
                    ux = np.concatenate([u_wall, u_surface, ux_random])
                    uy = np.concatenate([u_wall, u_surface, uy_random])
                    r_u = np.concatenate([r_u_wall, r_u_surface, r_u_random])
                r_ux = r_u
                r_uy = r_u
            else:
                r_u = r_train[index_for_train]
                r_ux = r_u
                r_uy = r_u
                ux, uy = ux_train[index_for_train], uy_train[index_for_train]
        else:
            r_u_wall, u_wall = self.generate_wall_points(u_num_x)
            # make surface ux at the cylinder
            r_u_surface = self.make_r_surface(u_num_surface, u_num_inner=u_num_inner)
            u_surface = np.zeros(len(r_u_surface))

            # velocity at inlet and outlet
            if self.use_inlet_outlet_u:
                r_u_inlet_outlet = self.make_r_mesh(
                    self.x_start,
                    self.x_end,
                    -self.w / 2 + self.slide,
                    self.w / 2 - self.slide,
                    2,
                    u_num_y,
                )
                # eta_u_inlet_outlet = self.x_to_eta(r_u_inlet_outlet[:, 0])
                # xi_u_inlet_outlet = self.y_to_xi(
                #     eta_u_inlet_outlet, r_u_inlet_outlet[:, 1])
                ux_inlet_outlet, uy_inlet_outlet = self.calc_u_v_4(r_u_inlet_outlet)
                # concatenate
                r_u = np.concatenate([r_u_inlet_outlet, r_u_wall])
                r_ux = r_u
                r_uy = r_u
                ux = np.concatenate([ux_inlet_outlet, u_wall])
                uy = np.concatenate([uy_inlet_outlet, u_wall])
            else:
                r_ux = np.concatenate([r_u_wall, r_u_surface])
                r_uy = np.concatenate([r_u_wall, r_u_surface])
                ux = np.concatenate([u_wall, u_surface])
                uy = np.concatenate([u_wall, u_surface])

        self.r += [r_ux, r_uy]
        self.f += [np.array(ux), np.array(uy)]

    def make_r_mesh_mixed(self, num_inner, num_outer, dr, pad):
        ratio = 0.3
        x_start = self.x_start + pad
        x_end = self.x_end - pad
        y_start = -self.w / 2 + pad
        y_end = self.w / 2 - pad
        num_x, num_y = self.num_to_num_x_y(
            num_outer, use_broad=self.use_broad_governing_eqs, ratio=ratio
        )
        r_f = self.make_r_mesh_sinusoidal(
            x_start, x_end, y_start, y_end, num_x, num_y, pad
        )
        r_circular = self.make_r_mesh_circular(num_inner, dr)

        index_out_dr = self.get_index_out_cylinder(r_f, self.particle_radius + dr)
        r_f = r_f[index_out_dr]
        r_f = np.concatenate([r_f, r_circular])
        return r_f

    def generate_f(self, f_num, f_pad, force=0.0, f_num_inner=0, dr=None):
        if f_num_inner:
            r_fx = self.make_r_mesh_mixed(f_num_inner, f_num, dr, f_pad)
            r_fy = r_fx
            # forceの値を設定
            if self.use_force_as_constant_pressure:
                force_x = -self.delta_p / self.L
            else:
                force_x = force
            force_y = force
            fx = np.full(len(r_fx), force_x)
            fy = np.full(len(r_fy), force_y)
            self.r += [r_fx, r_fy]
            self.f += [fx, fy]
        else:
            super().generate_f(f_num, f_pad, force=0.0)
            for i in range(1, 3):
                index_out_cylinder = self.get_index_out_cylinder(self.r[-i])
                self.r[-i] = self.r[-i][index_out_cylinder]
                self.f[-i] = self.f[-i][index_out_cylinder]

    def make_r_mesh_circular(self, num_per_side, dr=0.1):
        x_start = self.particle_center[0] - (self.particle_radius + dr)
        x_end = self.particle_center[0] + (self.particle_radius + dr)
        y_start = self.particle_center[1] - (self.particle_radius + dr)
        y_end = self.particle_center[1] + (self.particle_radius + dr)

        r = self.make_r_mesh(x_start, x_end, y_start, y_end, num_per_side, num_per_side)
        if self.__class__.__name__ == "SinusoidalCylinder":
            index_out_of_dr = self.get_index_out_cylinder(
                r, radius_min=self.particle_radius + dr * 1.1
            )
        elif self.__class__.__name__ == "SinusoidalRectangular":
            index_out_of_dr = self.get_index_out_cylinder(
                r, radius_min=self.particle_radius + dr
            )
        # index_in_dr = (r[:, 0] != r[index_out_of_dr][:, 0]) | (r[:, 1] != r[index_out_of_dr][:, 1])
        index_in_dr = np.arange(0, len(r), 1, dtype=int)
        index_in_dr = np.delete(index_in_dr, index_out_of_dr)
        bool_in_dr = np.zeros(len(r), dtype=bool)
        bool_in_dr[index_in_dr] = True
        index_out_of_radius = self.get_index_out_cylinder(r)
        bool_out_of_radius = np.zeros(len(r), dtype=bool)
        bool_out_of_radius[index_out_of_radius] = index_out_of_radius
        index_in_domain = bool_in_dr & bool_out_of_radius

        return r[index_in_domain]

    def generate_div(self, div_num, div_pad, divu=0.0, div_num_inner=0, dr=None):
        if div_num_inner:
            r_div = self.make_r_mesh_mixed(div_num_inner, div_num, dr, div_pad)
            div = np.full(len(r_div), divu)
            self.r += [r_div]
            self.f += [div]
        else:
            super().generate_div(div_num, div_pad, divu)
            index_out_cylinder = self.get_index_out_cylinder(self.r[-1])
            self.r[-1] = self.r[-1][index_out_cylinder]
            self.f[-1] = self.f[-1][index_out_cylinder]

    def generate_training_data(
        self,
        u_num_surface=None,
        u_num_wall=None,
        p_num=None,
        f_num=None,
        f_pad=None,
        div_num=None,
        div_pad=None,
        difu_num=None,
        difp_num=None,
        difp_loc="inlet_outlet",
        difp_pad=None,
        u_num_inner=5,
        dr=0.2,
        without_f=False,
        u_num_random=None,
        f_num_inner=0,
        div_num_inner=0,
    ):
        self.without_f = without_f
        self.r = []
        self.f = []
        self.generate_u(
            u_num_surface,
            u_num_wall,
            u_num_inner=u_num_inner,
            u_num_random=u_num_random,
        )
        self.generate_difu(difu_num)
        self.generate_f(f_num=f_num, f_pad=f_pad, f_num_inner=f_num_inner, dr=dr)
        self.generate_div(
            div_num=div_num, div_pad=div_pad, div_num_inner=div_num_inner, dr=dr
        )
        if self.use_difp:
            self.generate_difp(difp_num, difp_pad, difp_loc)
        return self.r, self.f

    def generate_test(
        self,
        test_num=None,
        infer_governing_eqs=False,
        infer_p=False,
        ux_test=False,
        use_fem_result=False,
        use_spm_result=True,
        infer_wall=False,
        infer_du_boundary=False,
        infer_du_grid=False,
    ):
        print(test_num)
        self.test_num = test_num

        if infer_wall:
            y_start = -(self.w / 2 + self.a) - 0.1
            y_end = -y_start
            num_y = self.test_num
            num_x = int(test_num * self.L / (y_end - y_start))
            r = self.make_r_mesh(self.x_start, self.x_end, y_start, y_end, num_x, num_y)
            ux_test, uy_test = self.calc_u_v_4(r)
            self.r_test = [r, r]
            self.f_test = [ux_test, uy_test]
            return self.r_test, self.f_test

        if self.infer_difp:
            difp_pad = 0.03
            y_start = -self.w / 2 + difp_pad
            y_end = self.w / 2 - difp_pad

            r_difp = self.make_r_mesh_sinusoidal(
                0.0,
                0.0,
                -self.w / 2 + self.slide,
                self.w / 2 - self.slide,
                1,
                test_num,
                self.slide,
            )

            difp = np.full(len(r_difp), self.delta_p)

            self.r_test = [r_difp]
            self.f_test = [difp]
            return self.r_test, self.f_test

        if use_spm_result:
            if self.particle_center[0] == 0.625:
                if test_num == 36:
                    filename = "0801_sinusoidalcylinder_test_64x36.pickle"
                elif test_num == 48:
                    filename = "0801_sinusoidalcylinder_test_86x48.pickle"
                elif test_num == 72:
                    filename = "0801_sinusoidalcylinder_test_128x72.pickle"
            elif self.particle_center[0] == 1.875:
                if self.particle_radius == 0.078125:
                    filename = "0802_sinusoidalcylinder_narrow_fixed_test_128x72.pickle"
                else:
                    if test_num == 48:
                        filename = "0801_sinusoidalcylinder_narrow_test_86x48.pickle"
                    elif test_num == 72:
                        filename = "0801_sinusoidalcylinder_narrow_test_128x72.pickle"
            print(filename)
            with open(
                # "/work/jh210017a/q24015/template_data/0314_cylinder_test_484.pickle",
                # "/work/jh210017a/q24015/template_data/0501_cylinder_25_test_484.pickle",
                f"{Path(os.path.abspath(__file__)).parent.parent}/template_data/test_sinusoidalcylinder_spm/{filename}",
                "rb",
            ) as file:
                save_dict = pickle.load(file)
            r, ux, uy = (
                save_dict["r"],
                save_dict["ux"] * 12,
                save_dict["uy"] * 12,
            )  # standarize

        if self.__class__.__name__ == "SinusoidalCylinder":
            class_name = "sinusoidalcylinder"
        elif self.__class__.__name__ == "SinusoidalRectangular":
            class_name = "sinusoidalrectangular"

        if use_fem_result:
            if self.particle_center[0] == 0.625:
                if test_num == 72:
                    filename = "fem_test_128x72.pickle"
                elif test_num == 36:
                    filename = "fem_test_64x36.pickle"
                elif test_num == 50:
                    filename = "fem_test_89x50.pickle"
                elif test_num == 54:
                    filename = "fem_test_96x54.pickle"
            with open(
                f"{Path(os.path.abspath(__file__)).parent.parent}/template_data/test_{class_name}_fem/{filename}",
                "rb",
            ) as file:
                save_dict = pickle.load(file)
            r, ux, uy = (
                save_dict["r"],
                save_dict["ux"],
                save_dict["uy"],
            )

        self.r_test += [r, r]
        self.f_test += [ux, uy]
        return self.r_test, self.f_test
