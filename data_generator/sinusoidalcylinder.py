import os
import pickle

import cmocean as cmo
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, vmap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy import integrate

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

    def make_r_surface(self, num, use_inner=False):
        θ_start = 0
        θ_end = 2 * np.pi
        θ = np.linspace(θ_start, θ_end, num)
        if not use_inner:
            r_mesh, θ_mesh = np.meshgrid(self.particle_radius, θ)
        else:
            radius_range = np.linspace(0.013, self.particle_radius, 3)
            r_mesh, θ_mesh = np.meshgrid(radius_range, θ)
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
        return r

    def get_index_out_cylinder(self, r, radius_min=None):
        rx = r[:, 0] - self.particle_center[0]
        ry = r[:, 1] - self.particle_center[1]
        if not radius_min:
            index_out_cylinder = np.where(
                rx**2 + ry**2 > self.particle_radius**2
            )[0]
        else:
            index_out_cylinder = np.where(rx**2 + ry**2 > radius_min**2)[0]
        return index_out_cylinder

    def get_index_in_channel(self, r):
        top_ok = r[:, 1] < self.calc_y_top(r[:, 0])
        bottom_ok = r[:, 1] > self.calc_y_bottom(r[:, 0])
        index_in_channel = top_ok & bottom_ok
        return index_in_channel

        # generation of training data

    def generate_u(self, u_num_surface, u_num_wall, u_b=0.0):
        """
        premise: ux,uy values are taken at same points
        """
        # split u_num
        u_num_x, u_num_y = self.num_to_num_x_y(u_num_wall)
        u_pad = self.slide

        if self.use_random_u:
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
            with open(
                f'{os.environ["HOME"]}/opt/stopro/template_data/test_from_fenics/0303_random_training_50.pickle',
                "rb",
            ) as file:
                save_dict = pickle.load(file)
            r_train = save_dict["r"]
            f_train = save_dict["u"]

            index_for_train = np.arange(0, len(r_train[0]), 1)
            rng = np.random.default_rng(seed=self.seed)
            rng.shuffle(index_for_train)
            index_for_train = index_for_train[:u_num_wall]
            r_u = r_train[0][index_for_train]
            r_ux = r_u
            r_uy = r_u
            ux, uy = f_train[0][index_for_train], f_train[1][index_for_train]
        else:
            r_u_wall, u_wall = self.generate_wall_points(u_num_x)
            # make surface ux at the cylinder
            r_u_surface = self.make_r_surface(u_num_surface)
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

    def generate_f(self, **kwargs):
        super().generate_f(**kwargs)
        for i in range(1, 3):
            index_out_cylinder = self.get_index_out_cylinder(self.r[-i])
            self.r[-i] = self.r[-i][index_out_cylinder]
            self.f[-i] = self.f[-i][index_out_cylinder]

    def generate_div(self, **kwargs):
        super().generate_div(**kwargs)
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
        num_inner=5,
        dr=0.2,
        without_f=False,
    ):
        self.without_f = without_f
        self.r = []
        self.f = []
        self.generate_u(u_num_surface, u_num_wall)
        self.generate_difu(difu_num)
        self.generate_f(f_num=f_num, f_pad=f_pad)
        self.generate_div(div_num=div_num, div_pad=div_pad)
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
    ):
        print(test_num)
        self.test_num = test_num
        if use_spm_result:
            if self.particle_center[0] == 0.625:
                if test_num == 36:
                    filename = "0801_sinusoidalcylinder_test_64x36.pickle"
                elif test_num == 48:
                    filename = "0801_sinusoidalcylinder_test_86x48.pickle"
                elif test_num == 72:
                    filename = "0801_sinusoidalcylinder_test_128x72.pickle"
            elif self.particle_center[0] == 1.875:
                if test_num == 48:
                    filename = "0801_sinusoidalcylinder_narrow_test_86x48.pickle"
                elif test_num == 72:
                    filename = "0801_sinusoidalcylinder_narrow_test_128x72.pickle"
            print(filename)
            with open(
                # "/work/jh210017a/q24015/template_data/0314_cylinder_test_484.pickle",
                # "/work/jh210017a/q24015/template_data/0501_cylinder_25_test_484.pickle",
                f'{os.environ["HOME"]}/opt/stopro/template_data/test_sinusoidalcylinder_spm/{filename}',
                "rb",
            ) as file:
                save_dict = pickle.load(file)
            r, ux, uy = (
                save_dict["r"],
                save_dict["ux"] * 12,
                save_dict["uy"] * 12,
            )  # standarize

        self.r_test += [r, r]
        self.f_test += [ux, uy]
        return self.r_test, self.f_test
