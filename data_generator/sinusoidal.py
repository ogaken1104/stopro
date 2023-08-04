import os
import pickle

import cmocean as cmo
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, vmap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy import integrate

from stopro.data_generator.stokes_2D_generator import StokesDataGenerator
from stopro.data_handler.data_handle_module import HdfOperator


class Sinusoidal(StokesDataGenerator):
    def __init__(
        self,
        oscillation_amplitude=None,
        channel_length=None,
        channel_width=None,
        slide=0.03,
        random_arrange=False,
        use_gradp_training=False,
        infer_gradp=False,
        infer_ux_wall=False,
        use_only_bottom_u=False,
        use_only_inlet_gradp=False,
        cut_last_x=False,
        use_difp=False,
        use_difu=False,
        use_inlet_outlet_u=False,
        x_start=0.0,
        use_random_u=False,
        delta_p=-30,
        use_force_as_constant_pressure=False,
        use_1d_u=False,
        seed=43,
        use_noisy_ux=False,
        noise_fraction=0.01,
        use_broad_governing_eqs=False,
        use_diff=False,
    ):
        super().__init__(random_arrange)
        self.a = oscillation_amplitude
        self.L = channel_length
        self.w = channel_width
        self.epsilon = self.w / self.L
        self.slide = slide
        self.use_difp = use_difp
        # self.delta_p = -12. * self.L / self.w  # need to confirm
        self.delta_p = delta_p
        self.Qp = -self.w * self.delta_p / (12 * self.L)
        self.gradpx = self.delta_p / self.L  # only satisfied in inlet and outlet
        self.gradpy = 0.0  # only satisfied in inlet and outlet
        self.Q0 = self.calc_Q0()
        self.Q2 = self.calc_Q2()
        self.r = []
        self.f = []
        self.r_test = []
        self.f_test = []
        self.infer_gradp = infer_gradp
        self.use_gradp_training = use_gradp_training
        self.use_only_inlet_gradp = use_only_inlet_gradp
        self.use_difu = use_difu
        self.use_inlet_outlet_u = use_inlet_outlet_u
        self.x_start = x_start
        self.x_end = self.x_start + self.L
        self.use_random_u = use_random_u
        self.use_force_as_constant_pressure = use_force_as_constant_pressure
        self.use_1d_u = use_1d_u
        self.seed = seed
        self.use_noisy_ux = use_noisy_ux
        self.noise_fraction = noise_fraction
        self.use_broad_governing_eqs = use_broad_governing_eqs
        self.use_diff = use_diff

    def calc_y_top(self, xs):
        return self.a * jnp.sin(2 * jnp.pi * xs / self.L) + self.w / 2

    def calc_y_bottom(self, xs):
        return -self.a * jnp.sin(2 * jnp.pi * xs / self.L) - self.w / 2

    def calc_Q0(self):
        return ((1 - 4 * self.a**2) ** (5 / 2)) / (1 + 2 * self.a**2)

    def calc_Q2(self):
        first = 12 * jnp.pi**2 * self.a**2 / 5
        second = ((1 - 4 * self.a**2) ** (7 / 2)) / ((1 + 2 * self.a**2) ** 2)
        return -first * second

    def h(self, eta):
        return 1 / 2 + self.a * jnp.sin(2 * jnp.pi * eta)

    def calc_u0(self, eta, xi):
        numerator = 3 * self.Q0
        denominator = 2 * (1 + 2 * self.a * jnp.sin(2 * jnp.pi * eta))
        return numerator / denominator * (1 - xi**2)

    def calc_v0(self, eta, xi):
        numerator = 3 * self.Q0 * jnp.pi * self.a * jnp.cos(2 * jnp.pi * eta)
        denominator = self.L * (1 + 2 * self.a * jnp.sin(2 * jnp.pi * eta))
        return numerator / denominator * (xi - xi**3)

    def calc_u2(self, eta, xi):
        coefficient = 2 / (1 + 2 * self.a * jnp.sin(2 * jnp.pi * eta))
        first_1 = 3 / 20 * self.Q0 * jnp.pi**2 * self.a
        first_2 = (
            2 * self.a
            + jnp.sin(2 * jnp.pi * eta)
            + 6 * self.a * jnp.cos(2 * jnp.pi * eta) ** 2
        )
        first_3 = 5 * xi**4 - 6 * xi**2 + 1
        second = 3 / 4 * self.Q2 * (1 - xi**2)
        return coefficient * (first_1 * first_2 * first_3 + second)

    def calc_v2(self, eta, xi):
        coeeficient = 6 * jnp.pi * self.a * jnp.cos(2 * jnp.pi * eta) / self.L
        first_coef = jnp.pi**2 * self.Q0 / 20
        first_1_1 = 12 * self.a * jnp.sin(2 * jnp.pi * eta) - 1
        first_1_2 = xi**5 - 2 * xi**3 + xi
        first_2_1 = 2 * self.a / (1 + 2 * self.a * jnp.sin(2 * jnp.pi * eta))
        first_2_2 = (
            2 * self.a
            + jnp.sin(2 * jnp.pi * eta)
            + 6 * self.a * jnp.cos(2 * jnp.pi * eta) ** 2
        )
        first_2_3 = 5 * xi**5 - 6 * xi**3 + xi
        second_1 = (
            3 * self.a * self.Q2 / 2 / (1 + 2 * self.a * jnp.sin(2 * jnp.pi * eta))
        )
        second_2 = -(xi**3) + xi
        first = first_coef * (first_1_1 * first_1_2 + first_2_1 * first_2_2 * first_2_3)
        second = second_1 * second_2
        return coeeficient * (first + second)

    ############### for calculationg 4th term ###############
    def h_eta1(self, eta):
        return 2 * jnp.pi * self.a * jnp.cos(2 * jnp.pi * eta)

    def h_eta2(self, eta):
        return -((2 * jnp.pi) ** 2) * self.a * jnp.sin(2 * jnp.pi * eta)

    def h_eta3(self, eta):
        return -((2 * jnp.pi) ** 3) * self.a * jnp.cos(2 * jnp.pi * eta)

    def h_eta4(self, eta):
        return (2 * jnp.pi) ** 4 * self.a * jnp.sin(2 * jnp.pi * eta)

    def calc_Q4(self):
        """Compute Q4 using numerical integration"""
        integrand = self.Q4_integrand
        integral, _abs_error = integrate.quad(integrand, 0.0, 1.0)
        #         print(f'integral={integral}')
        Q0 = self.calc_Q0()
        Q2 = self.calc_Q2()
        return Q2**2 / Q0 - Q0**2 / 1400 * integral

    def Q4_integrand(self, eta):
        _h = self.h(eta)
        term1 = 87 * self.h_eta1(eta) ** 4
        term2 = -306 * _h * self.h_eta1(eta) ** 2 * self.h_eta2(eta)
        term3 = 54 * _h**2 * self.h_eta2(eta) ** 2
        term4 = 56 * _h**2 * self.h_eta1(eta) * self.h_eta3(eta)
        term5 = -4 * _h**3 * self.h_eta4(eta)
        return (term1 + term2 + term3 + term4 + term5) / _h**3

    def calc_Q(self):
        return (
            self.calc_Q0()
            + self.epsilon**2 * self.calc_Q2()
            + self.epsilon**4 * self.calc_Q4()
        )
        # return self.calc_Q0() + self.epsilon**2 * self.calc_Q2()

    def calc_psi_4(self, x, y):
        eta = self.x_to_eta(x)
        xi = self.y_to_xi(eta, y)
        Q0 = self.calc_Q0()
        Q2 = self.calc_Q2()
        Q4 = self.calc_Q4()
        _h = self.h(eta)
        _h1 = self.h_eta1(eta)
        _h2 = self.h_eta2(eta)
        _h3 = self.h_eta3(eta)
        _h4 = self.h_eta4(eta)

        term1 = Q4 * (3 / 4 * xi - 1 / 4 * xi**3)
        term2 = -Q2 * 3 / 40 * (_h * _h2 - 4 * _h1**2) * xi * (1 - xi**2) ** 2
        # calculate term3
        first = (408 + 1800 * xi**2) * _h1**4
        second = (684 - 1800 * xi**2) * _h * _h1**2 * _h2
        third = -(270 - 180 * xi**2) * _h**2 * _h2**2
        fourth = -(248 - 240 * xi**2) * _h**2 * _h1 * _h3
        fifth = (19 - 15 * xi**2) * _h**3 * _h4
        term3 = (
            -Q0
            / 5600
            * (first + second + third + fourth + fifth)
            * xi
            * (1 - xi**2) ** 2
        )
        return term1 + term2 + term3

    def outermap(self, f):
        #             return vmap(vmap(f, in_axes=(None, 0)), in_axes=(0, None))
        return vmap(f, in_axes=(0, 0))

    def calc_u4(self, x, y):
        psi = self.calc_psi_4

        def _dpsi_dy(x, y):
            return grad(psi, 1)(x, y)

        dpsi_dy = self.outermap(_dpsi_dy)
        return dpsi_dy(x, y)

    def calc_v4(self, x, y):
        psi = self.calc_psi_4

        def _dpsi_dx(x, y):
            return grad(psi, 0)(x, y)

        dpsi_dx = self.outermap(_dpsi_dx)
        return -dpsi_dx(x, y)

    def calc_u_v_4(self, r):
        eta = self.x_to_eta(r[:, 0])
        xi = self.y_to_xi(eta, r[:, 1])
        # longitudinal velocity
        u0 = self.calc_u0(eta, xi)
        u2 = self.calc_u2(eta, xi)
        u4 = self.calc_u4(r[:, 0], r[:, 1])
        # transverse velocity
        v0 = self.calc_v0(eta, xi)
        v2 = self.calc_v2(eta, xi)
        v4 = self.calc_v4(r[:, 0], r[:, 1])
        u = u0 + self.epsilon**2 * u2 + self.epsilon**4 * u4
        v = v0 + self.epsilon**2 * v2 + self.epsilon**4 * v4
        return np.array(u) * self.Qp, np.array(v) * self.Qp

    ########################################################################################

    def calc_u_v(self, eta, xi):
        # longitudinal velocity
        u0 = self.calc_u0(eta, xi)
        u2 = self.calc_u2(eta, xi)
        # transverse velocity
        v0 = self.calc_v0(eta, xi)
        v2 = self.calc_v2(eta, xi)
        u = u0 + self.epsilon**2 * u2
        v = v0 + self.epsilon**2 * v2
        return u * self.Qp, v * self.Qp

    def x_to_eta(self, x):
        return self.epsilon * x

    def y_to_xi(self, eta, y):
        return y / (self.h(eta))

    def num_to_num_x_y(self, num, use_broad=False, ratio=None):
        if use_broad:
            height = (self.w + self.a * 2) * (1 + ratio)
            length = self.L * (1 + ratio)
            # adjust = height / self.w  # use_broadがFalseの場合と点の間隔が（大体）同じになるように
            adjust = 1
            num_x = int(num * self.L / (height + self.L) * adjust * (1 + ratio))
            num_y = int(num * height / (height + self.L) * adjust * (1 + ratio))
        else:
            num_x = int(num * self.L / (self.w + self.L))
            num_y = int(num * self.w / (self.w + self.L))
        return num_x, num_y

    def change_outside_values_to_zero(self, r, f):
        eta_u = self.x_to_eta(r[:, 0])
        xi_u = self.y_to_xi(eta_u, r[:, 1])
        is_outside_the_domain = np.where(
            (r[:, 1] > self.calc_y_top(r[:, 0]))
            | (r[:, 1] < self.calc_y_bottom(r[:, 0]))
        )
        f[is_outside_the_domain] = 0.0
        return f

    def generate_wall_points(self, num_x):
        # velocity at wall
        x_wall_half = np.linspace(self.x_start, self.x_end, num_x)
        x_wall = np.concatenate([x_wall_half, x_wall_half])
        y_wall = np.concatenate(
            [self.calc_y_top(x_wall_half), self.calc_y_bottom(x_wall_half)]
        )
        r_wall = np.stack([x_wall, y_wall], axis=1)
        f_wall = np.zeros(len(r_wall))
        return r_wall, f_wall

    # generation of training data
    def generate_u(self, u_num, u_b=0.0):
        """
        premise: ux,uy values are taken at same points
        """
        # split u_num
        u_num_x, u_num_y = self.num_to_num_x_y(u_num)
        u_pad = self.slide

        if self.use_random_u:
            # ############ when using semi-analytical ##############
            # u_num_x, u_num_y = self.num_to_num_x_y(28)
            # x_start = self.x_start+u_pad
            # x_end = self.x_end-u_pad
            # y_start = -self.w/2 + u_pad
            # y_end = self.w/2 - u_pad
            # r_u_all = self.make_r_mesh_sinusoidal(
            #     x_start, x_end, y_start, y_end, int(u_num_x), int(u_num_y), u_pad)
            # rng = np.random.default_rng(seed=42)
            # rng.shuffle(r_u_all, axis=0)
            # r_u = r_u_all[:u_num]
            # # eta_u = self.x_to_eta(r_u[:, 0])
            # # xi_u = self.y_to_xi(eta_u, r_u[:, 1])
            # ux, uy = self.calc_u_v_4(r_u)

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
            index_for_train = index_for_train[:u_num]
            r_u = r_train[0][index_for_train]
            r_ux = r_u
            r_uy = r_u
            ux, uy = f_train[0][index_for_train], f_train[1][index_for_train]

        elif self.use_1d_u:
            # with open('/work/jh210017a/q24015/template_data/test_from_fenics/0303_1d_training_50.pickle', 'rb') as file:
            #     save_dict = pickle.load(file)
            # r_train = save_dict['r']
            # f_train = save_dict['u']
            # r_ux, r_uy = r_train
            # ux, uy = f_train

            # split = 4
            # if split:
            #     r_ux = r_ux[::split]
            #     r_uy = r_uy[::split]
            #     ux = ux[::split]
            #     ux = uy[::split]
            ########## when using fem as reference solution #################
            with open(
                f'{os.environ["HOME"]}/opt/stopro/template_data/test_from_fenics/0303_random_training_50.pickle',
                "rb",
            ) as file:
                save_dict = pickle.load(file)
            r_train = save_dict["r"]
            f_train = save_dict["u"]
            r_ux_all, r_uy_all = r_train
            ux_all, uy_all = f_train
            split = 4
            num_lines = 2
            if num_lines == 4:
                # points for ux
                xs = r_ux_all[:, 0]
                xs_unique = np.sort(np.unique(xs))
                ## 4line case ##
                index_x = 15
                index2_x = 21
                xs_for_ux = np.array([xs_unique[index_x], xs_unique[-index_x]])

                r_ux = np.empty((0, 2))
                ux = np.empty(0)
                for x_for_ux in xs_for_ux:
                    index_ux = np.isin(xs, x_for_ux)
                    r_ux = np.append(r_ux, r_ux_all[index_ux][::split], axis=0)
                    ux = np.append(ux, ux_all[index_ux][::split])

                # points for uy
                ys = r_uy_all[:, 1]
                ys_unique = np.sort(np.unique(ys))
                # 4line case ##
                index_y = 2
                index2_y = 10
                ys_for_uy = np.array(
                    [
                        ys_unique[index_y],
                        ys_unique[index2_y],
                        ys_unique[-index2_y],
                        ys_unique[-index_y],
                    ]
                )

                r_uy = np.empty((0, 2))
                uy = np.empty(0)
                for y_for_uy in ys_for_uy:
                    index_uy = np.isin(ys, y_for_uy)
                    r_uy = np.append(r_uy, r_uy_all[index_uy][::split], axis=0)
                    uy = np.append(uy, uy_all[index_uy][::split])
            elif num_lines == 2:
                split = 2
                # points for ux
                index_y = 16
                ys = r_ux_all[:, 1]
                ys_unique = np.sort(np.unique(ys))
                ys_for_ux = np.array([ys_unique[index_y]])
                r_ux = np.empty((0, 2))
                ux = np.empty(0)
                for y_for_ux in ys_for_ux:
                    index_ux = np.isin(ys, y_for_ux)
                    r_ux = np.append(r_ux, r_ux_all[index_ux][::split], axis=0)
                    ux = np.append(ux, ux_all[index_ux][::split])

                # points for uy
                index_x = 30
                xs = r_uy_all[:, 0]
                xs_unique = np.sort(np.unique(xs))
                xs_for_uy = np.array([xs_unique[index_x]])
                r_uy = np.empty((0, 2))
                uy = np.empty(0)
                for x_for_uy in xs_for_uy:
                    index_uy = np.isin(xs, x_for_uy)
                    r_uy = np.append(r_uy, r_uy_all[index_uy][::split], axis=0)
                    uy = np.append(uy, uy_all[index_uy][::split])

                ## if using same 2 lines for ux and uy
                index_x = np.isin(xs, xs_for_uy[0])
                index_y = np.isin(ys, ys_for_ux[0])
                index = index_x | index_y
                # print(index, len(r_ux_all), len(index))
                r_ux = r_ux_all[index][::split]
                r_uy = r_uy_all[index][::split]
                ux = ux_all[index][::split]
                uy = uy_all[index][::split]

                # if also using wall points
                r_u_wall, u_wall = self.generate_wall_points(u_num_x)
                r_ux = np.concatenate([r_ux, r_u_wall])
                r_uy = np.concatenate([r_uy, r_u_wall])
                print(r_ux, r_ux.shape)
                ux = np.concatenate([ux, u_wall])
                uy = np.concatenate([uy, u_wall])

        else:
            # velocity at wall
            # x_u_wall_half = np.linspace(self.x_start, self.x_end, u_num_x)
            # x_u_wall = np.concatenate([x_u_wall_half, x_u_wall_half])
            # y_u_wall = np.concatenate(
            #     [self.calc_y_top(x_u_wall_half), self.calc_y_bottom(x_u_wall_half)]
            # )
            # r_u_wall = np.stack([x_u_wall, y_u_wall], axis=1)
            # u_wall = np.zeros(len(r_u_wall))
            r_u_wall, u_wall = self.generate_wall_points(u_num_x)

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
                r_ux = r_u_wall
                r_uy = r_u_wall
                ux = u_wall
                uy = u_wall

        if self.use_noisy_ux:
            # データの読み込み
            try:
                with open(
                    f'{os.environ["HOME"]}/opt/stopro/template_data/test_from_fenics/0303_interpolation_50.pickle',
                    "rb",
                ) as file:
                    save_dict = pickle.load(file)
            except:
                with open(
                    f'{os.environ["HOME"]}/template_data/test_from_fenics/0303_interpolation_50.pickle',
                    "rb",
                ) as file:
                    save_dict = pickle.load(file)
            ux_max = np.max(save_dict["ux"])
            ux_min = np.min(save_dict["ux"])
            # print(ux_max, ux_min)
            std_noise = (
                ux_max - ux_min
            ) * self.noise_fraction  # （最大値-最小値）*noise_fractionの大きさの標準偏差を用いてノイズを生成する
            noise_ux = np.random.normal(0.0, std_noise, len(ux))
            print(ux)
            ux += noise_ux
            print(ux)

        self.r += [r_ux, r_uy]
        self.f += [np.array(ux), np.array(uy)]

    def generate_test(
        self,
        test_num=0,
        ux_test=False,
        infer_p=False,
        use_spm_result=False,
        use_fem_result=False,
        infer_governing_eqs=False,
        infer_wall=False,
        infer_du_boundary=False,
        infer_du_grid=False,
    ):
        self.test_num = test_num
        if infer_du_boundary:
            num_x = 2000
            pad_boundary = 0.0  # 0.00001
            x = np.linspace(0.0, self.L, num_x)
            y_top, y_bottom = self.calc_y_top(x), self.calc_y_bottom(x)
            y = np.concatenate([y_top - pad_boundary, y_bottom + pad_boundary])
            r = np.stack([np.concatenate([x, x]), y], axis=1)
            duidui = np.zeros(len(r))
            self.r_test = [r] * 4
            self.f_test = [duidui] * 4
            return self.r_test, self.f_test

        if infer_governing_eqs:
            ## lattice alignment version
            maximum_height = self.w / 2 + self.a
            num_y = self.test_num
            num_x = int(test_num * self.L / (maximum_height * 2))
            r = self.make_r_mesh(
                self.x_start, self.x_end, -maximum_height, maximum_height, num_x, num_y
            )

            # ## 1d alignment version
            # x_widest = 0.59933333
            # x_narrowest = 1.8193333333333335
            # # x_widest = 0.625
            # # x_narrowest = 1.875
            # num_y = 100

            # y_start_widest, y_start_narrowest = self.calc_y_bottom(
            #     np.array([x_widest, x_narrowest])
            # )
            # y_end_widest, y_end_narrowest = self.calc_y_top(
            #     np.array([x_widest, x_narrowest])
            # )
            # r_widest = self.make_r_mesh(
            #     x_widest, x_widest, y_start_widest, y_end_widest, 1, num_y
            # )
            # r_narrowest = self.make_r_mesh(
            #     x_narrowest, x_narrowest, y_start_narrowest, y_end_narrowest, 1, num_y
            # )
            # r = np.concatenate([r_widest, r_narrowest])

            ## common part
            force_test = np.zeros(len(r))
            div_test = np.zeros(len(r))
            self.r_test = [r, r, r]
            self.f_test = [force_test, force_test, div_test]
            return self.r_test, self.f_test

        if infer_wall:
            y_start = 0.2
            y_end = 0.8
            num_y = self.test_num
            num_x = int(test_num * self.L / (y_end - y_start))
            r = self.make_r_mesh(self.x_start, self.x_end, y_start, y_end, num_x, num_y)
            ux_test, uy_test = self.calc_u_v_4(r)
            self.r_test = [r, r]
            self.f_test = [ux_test, uy_test]
            return self.r_test, self.f_test

        if use_spm_result:
            hdf_operator = HdfOperator(
                f'{os.environ["HOME"]}/opt/stopro/template_data/test_sinusoidal_from_spm_adimentionalized'
            )
            r_test, f_test = hdf_operator.load_test_data(["r", "f"], ["ux", "uy"])
            self.r_test = r_test
            self.f_test = f_test
            return self.r_test, self.f_test

        if use_fem_result:
            try:
                if test_num == 18:
                    file_name = "0303_interpolation_50.pickle"
                elif test_num == 36:
                    file_name = "0508_interpolation_num_36.pickle"
                elif test_num == 54:
                    file_name = "0508_interpolation_num_54.pickle"
                with open(
                    f'{os.environ["HOME"]}/opt/stopro/template_data/test_from_fenics/{file_name}',
                    "rb",
                ) as file:
                    save_dict = pickle.load(file)
            except:
                with open(
                    f'{os.environ["HOME"]}/template_data/test_from_fenics/0303_interpolation_50.pickle',
                    "rb",
                ) as file:
                    save_dict = pickle.load(file)
            r = save_dict["r"]
            self.r_test = [r, r]
            self.f_test = [save_dict["ux"], save_dict["uy"]]
            return self.r_test, self.f_test
            # ##### checking values at x from FEniCS in 1D
            # with open('/work/jh210017a/q24015/template_data/test_from_fenics/0306_1d_test_velocity.pickle', 'rb') as file:
            #     save_dict = pickle.load(file)
            # r_ux, r_uy = save_dict['r']
            # ux, uy = save_dict['u']
            # self.r_test = [r_ux, r_uy]
            # self.f_test = [ux, uy]
            # return self.r_test, self.f_test

        if not ux_test:
            # num_x, num_y = self.num_to_num_x_y(test_num)
            # r = self.make_r_mesh_sinusoidal(
            #     0., self.L, -self.w/2, self.w/2, num_x, num_y, 0)
            # r_ux = r
            # r_uy = r
            # # calc ux_test, uy_test
            # eta_u = self.x_to_eta(r[:, 0])
            # xi_u = self.y_to_xi(eta_u, r[:, 1])
            # ux_test, uy_test = self.calc_u_v(eta_u, xi_u)
            # if not self.infer_gradp:
            #     if not p_infer:
            #         r_p = np.array([[0., 0.]])
            #     elif p_infer:
            #         # r_p = self.make_r_mesh_sinusoidal(
            #         #     0., self.L, -self.w/2, self.w/2, 2, int(test_num/2), 0, periodic_test=True
            #         # )
            # elif self.infer_gradp:
            #     px_test = np.full(len(r), 0)
            #     py_test = np.full(len(r), 0)

            # lattice alignment version
            maximum_height = self.w / 2 + self.a
            num_y = test_num
            num_x = int(test_num * self.L / (maximum_height * 2))
            r = self.make_r_mesh(
                self.x_start, self.x_end, -maximum_height, maximum_height, num_x, num_y
            )
            #######################

            # # checking values at x = 0.625, 1.875
            # num_y = 100
            # x1 = 0.625
            # x2 = 1.875
            # x3 = 0.9375
            # x4 = 1.5625
            # r1 = self.make_r_mesh(x1, x2, -maximum_height,
            #                       maximum_height, 2, num_y)
            # r2 = self.make_r_mesh(x3, x4, -maximum_height,
            #                       maximum_height, 2, num_y)
            # r = np.concatenate([r1, r2])
            # # ########################

            r_ux = r
            r_uy = r
            r_p = r
            # calc ux_test, uy_test
            #             eta_u = self.x_to_eta(r[:, 0])
            #             xi_u = self.y_to_xi(eta_u, r[:, 1])
            ux_test, uy_test = self.calc_u_v_4(r)
            is_outside_the_domain = np.where(
                (r[:, 1] > self.calc_y_top(r[:, 0]))
                | (r[:, 1] < self.calc_y_bottom(r[:, 0]))
            )
            ux_test[is_outside_the_domain] = 0.0
            uy_test[is_outside_the_domain] = 0.0
            if self.use_gradp_training:
                if not self.infer_gradp:
                    r_p = np.array([[self.x_start, 0.0]])
                    p_test = np.zeros(len(r_p))
                elif self.infer_gradp:
                    r_p = self.make_r_mesh(
                        self.x_start,
                        self.x_end,
                        -self.w / 2 + self.slide,
                        self.w / 2 - self.slide,
                        2,
                        int(test_num / 2),
                    )
                    print(r_p, test_num)
                    p_test = np.zeros(test_num)
                    p_test[r_p[:, 0] == self.x_start] = -self.delta_p / 2
                    p_test[r_p[:, 0] == self.x_end] = self.delta_p / 2
                    px_test = np.full(len(r_p), self.gradpx)
                    py_test = np.full(len(r_p), self.gradpy)
            else:
                if not infer_p:
                    # r_p = np.array([[0., 0.]])
                    # p_test = np.zeros(len(r_p))
                    pass
                elif infer_p:
                    p_test = np.zeros(len(r_p))
                    p_test[r_p[:, 0] == self.x_start] = -self.delta_p / 2
                    p_test[r_p[:, 0] == self.x_end] = self.delta_p / 2

            # p_test = np.zeros(len(r_p))
        elif ux_test:
            r_ux = self.make_r_mesh(self.x_start, self.x_end, 0.0, 1.0, 5, 100)
            ux_test = self.f_ux(r_ux)
            r_uy = np.array([[self.x_start, 0.0]])
            uy_test = np.zeros(len(r_uy))
            r_p = np.array([[self.x_start, 0.0]])
            p_test = self.f_p(r_p)

        if self.use_gradp_training:
            if self.infer_gradp:
                self.r_test = [r, r, r_p, r_p]
                self.f_test = [ux_test, uy_test, px_test, py_test]
            else:
                self.r_test = [r_ux, r_uy]
                self.f_test = [ux_test, uy_test]
        else:
            if infer_p:
                self.r_test = [r, r, r_p]
                self.f_test = [ux_test, uy_test, p_test]
            else:
                self.r_test = [r, r]
                self.f_test = [ux_test, uy_test]

        return self.r_test, self.f_test

    def generate_p(self, p_num):
        p_num_x = 2
        p_num_y = int(p_num / p_num_x)
        r_p = np.array([[self.x_start, 0.0]])
        p = np.array([0.0])
        # r_p = self.make_r_mesh(
        #     0., self.L, -self.w/2+self.slide, self.w/2-self.slide, p_num_x, p_num_y)
        # p = np.zeros(p_num)
        # p[r_p[:, 0] == 0.] = -self.delta_p/2
        # p[r_p[:, 0] == self.L] = self.delta_p/2

        # p = np.concatenate([np.full(p_num_y, -self.delta_p),
        #                    np.full(p_num_y,  0.)])
        self.r += [r_p]
        self.f += [p]

    def generate_f(self, f_num, f_pad, force=0.0):
        """
        premise: ux,uy values are taken at same points
        """
        x_start = self.x_start + f_pad
        x_end = self.x_end - f_pad
        y_start = -self.w / 2 + f_pad
        y_end = self.w / 2 - f_pad
        # ## use broad along x
        # x_start -= 0.2
        # x_end += 0.2

        ratio = 0.3
        f_num_x, f_num_y = self.num_to_num_x_y(
            f_num, use_broad=self.use_broad_governing_eqs, ratio=ratio
        )
        if self.use_broad_governing_eqs:
            x_start -= self.L * ratio / 2
            x_end += self.L * ratio / 2
            y_start -= (self.w + self.a * 2) * ratio / 2
            y_end += (self.w + self.a * 2) * ratio / 2

            r_fx = self.make_r_mesh(
                x_start, x_end, y_start - self.a, y_end + self.a, f_num_x, f_num_y
            )
            r_fy = r_fx
            # forceの値を設定
            if self.use_force_as_constant_pressure:
                force_x = -self.delta_p / self.L
            else:
                force_x = force
            force_y = force
            fx = np.full(len(r_fx), force_x)
            fy = np.full(len(r_fy), force_y)
            fx = self.change_outside_values_to_zero(r_fx, fx)
        else:
            if self.random_arrange:
                pass
            else:
                r_fx = self.make_r_mesh_sinusoidal(
                    x_start, x_end, y_start, y_end, f_num_x, f_num_y, f_pad
                )
                r_fy = self.make_r_mesh_sinusoidal(
                    x_start, x_end, y_start, y_end, f_num_x, f_num_y, f_pad
                )

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

    def generate_div(self, div_num, div_pad, divu=0.0):
        x_start = self.x_start + div_pad
        x_end = self.x_end - div_pad
        y_start = -self.w / 2 + div_pad
        y_end = self.w / 2 - div_pad

        div_num_x, div_num_y = self.num_to_num_x_y(div_num)
        if self.random_arrange:
            pass
        else:
            r_div = self.make_r_mesh_sinusoidal(
                x_start, x_end, y_start, y_end, div_num_x, div_num_y, div_pad
            )
        div = np.full(len(r_div), divu)
        # div = np.full(len(r_div), divu)
        self.r += [r_div]
        self.f += [div]

    def generate_difp(self, difp_num, difp_pad, difp_loc):
        # use difp all inside
        x_start = -self.L / 8 + difp_pad
        x_end = self.L / 8 - difp_pad
        y_start = -self.w / 2 + difp_pad
        y_end = self.w / 2 - difp_pad
        difp_num_x, difp_num_y = self.num_to_num_x_y(difp_num)
        if difp_loc == "all_inside":
            r_difp = self.make_r_mesh_sinusoidal(
                x_start,
                x_end,
                y_start,
                y_end,
                int(difp_num_x / 4),
                difp_num_y,
                difp_pad,
            )
        elif difp_loc == "inlet_outlet":
            # use difp on inlet and outlet
            r_difp = self.make_r_mesh_sinusoidal(
                0.0,
                0.0,
                -self.w / 2 + self.slide,
                self.w / 2 - self.slide,
                1,
                difp_num,
                self.slide,
            )
        elif difp_loc == "1pair":
            r_difp = np.array([[0.0, 0.0]])

        if self.use_force_as_constant_pressure:
            difference_p = 0.0
        else:
            difference_p = self.delta_p
        difp = np.full(len(r_difp), difference_p)
        self.r += [r_difp]
        self.f += [difp]

    def generate_gradp(self, p_num):
        if not self.use_only_inlet_gradp:
            num_along_x = 2
        else:
            num_along_x = 1
        p_num = int(p_num / num_along_x)
        r_gradpx = self.make_r_mesh_sinusoidal(
            self.x_start,
            self.x_end,
            -self.w / 2 + self.slide,
            self.w / 2 - self.slide,
            num_along_x,
            p_num,
            self.slide,
        )
        r_gradpy = self.make_r_mesh_sinusoidal(
            self.x_start,
            self.x_end,
            -self.w / 2 + self.slide,
            self.w / 2 - self.slide,
            num_along_x,
            p_num,
            self.slide,
        )
        # r_gradpy[r_gradpy[:, 0] == self.x_start] += np.array([self.L/4, 0])
        # r_gradpy[r_gradpy[:, 0] == self.x_end] -= np.array([self.L/4, 0])
        gradpx = np.full(len(r_gradpx), self.gradpx)
        gradpy = np.full(len(r_gradpy), self.gradpy)
        self.r += [r_gradpx, r_gradpy]
        self.f += [gradpx, gradpy]

    def generate_training_data(
        self,
        u_num,
        p_num,
        f_num,
        f_pad,
        div_num,
        div_pad,
        difp_num=None,
        difp_pad=None,
        difp_loc="all_inside",
        difu_num=None,
        diff_num=None,
        without_f=False,
    ):
        self.without_f = without_f
        if without_f:
            self.generate_u(u_num)
            self.generate_difu(difu_num)
            self.generate_div(div_num, div_pad)
            self.generate_difp(difp_num, difp_pad, difp_loc)
        else:
            self.r, self.f = super().generate_training_data(
                u_num,
                p_num,
                f_num,
                f_pad,
                div_num,
                div_pad,
                difp_num,
                difp_pad,
                difp_loc,
                difu_num,
                diff_num,
            )
            # if self.use_difp:
            #     self.generate_difp(difp_num, difp_pad, difp_loc)
        return self.r, self.f

    def make_r_mesh_sinusoidal(
        self, x_start, x_end, y_start, y_end, num_x, num_y, pad, periodic_test=False
    ):
        y_start = self.calc_y_bottom(x_start)
        y_end = self.calc_y_top(x_end)
        interval_y = (y_end - y_start) / (num_y - 1)
        _r_div_x = np.linspace(x_start, x_end, num_x)
        y_max = self.calc_y_top(_r_div_x) - pad
        y_min = self.calc_y_bottom(_r_div_x) + pad
        r_div = np.empty(0)
        for i in range(len(_r_div_x)):
            _num_y = int((y_max[i] - y_min[i]) / interval_y) + 1
            r_div_y = np.linspace(y_min[i], y_max[i], _num_y)
            xx, yy = np.meshgrid(_r_div_x[i], r_div_y)
            _r_div = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
            if i == 0:
                r_div = _r_div
                _r_div_0 = _r_div
            else:
                # 最も右の点の座標をx=0と同じにする
                if i == len(_r_div_x) - 1 and periodic_test:
                    _r_div = _r_div_0.copy()
                    _r_div[:, 0] = self.x_end
                r_div = np.concatenate([r_div, _r_div])
        return r_div

    def plot_test(
        self,
        val_limits=[[0.0, 1.0], [0.0, 1.0], [0.0, 1]],
        save=False,
        path=None,
        show=False,
    ):
        fig, axs = plt.subplots(figsize=(5 * 2, 3), ncols=2, sharex=True, sharey=True)
        clrs = [self.COLOR["mid"], self.COLOR["mid"], self.COLOR["superfine"]]
        lbls = ["ux", "uy", "p"]
        xs = np.linspace(self.x_start, self.x_end, 100)
        y_top = self.calc_y_top(xs)
        y_bottom = self.calc_y_bottom(xs)
        axes = axs.reshape(-1)
        cmaps = [cmo.cm.dense, cmo.cm.balance, cmo.cm.dense]

        maximum_height = self.w / 2 + self.a
        y_num = self.test_num
        x_num = len(self.r_test[0][::y_num])
        y_grid = np.linspace(-maximum_height, maximum_height, y_num)
        x_grid = np.linspace(self.x_start, self.x_end, x_num)
        for i, ax in enumerate(axes):
            ax.plot(xs, y_top, color="k")
            ax.plot(xs, y_bottom, color="k")
            if i == 0 or i == 1:
                f_for_plot = self.change_outside_values_to_zero(
                    self.r_test[i], self.f_test[i]
                )
                f_mesh = f_for_plot.reshape(y_num, x_num)
                mappable = ax.pcolormesh(
                    x_grid,
                    y_grid,
                    f_mesh,
                    cmap=cmaps[i],
                    vmin=val_limits[i][0],
                    vmax=val_limits[i][1],
                    shading="nearest",
                )
            elif i == 2:
                mappable = ax.scatter(
                    self.r_test[i][:, 0],
                    self.r_test[i][:, 1],
                    c=self.f_test[i],
                    marker="o",
                    s=25.0,
                    cmap=cmaps[i],
                    vmin=val_limits[i][0],
                    vmax=val_limits[i][1],
                )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(mappable, cax=cax)
            ax.set_title(lbls[i])
            # ax.set_aspect('equal', adjustable='box')
            if self.__class__.__name__ == "SinusoidalCylinder":
                if i == 0:
                    num_surface = 100
                    r_surface = self.make_r_surface(num_surface)
                ax.plot(r_surface[:, 0], r_surface[:, 1], color="k")
        fig.tight_layout()
        if show:
            plt.show()
        if save:
            dir_path = f"{path}/fig"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            fig.savefig(f"{dir_path}/test.png")
        plt.clf()
        plt.close()

    def plot_train(self, save=False, path=None, show=False):
        COLORS = [
            "#DCBCBC",
            "#C79999",
            "#B97C7C",
            "#A25050",
            "#8F2727",
            "#7C0000",
            "#DCBCBC20",
            "#8F272720",
            "#00000060",
        ]
        COLOR = {
            i[0]: i[1]
            for i in zip(
                [
                    "light",
                    "light_highlight",
                    "mid",
                    "mid_highlight",
                    "dark",
                    "dark_highlight",
                    "light_trans",
                    "dark_trans",
                    "superfine",
                ],
                COLORS,
            )
        }
        if self.use_gradp_training and self.use_difu:
            fig, axs = plt.subplots(
                figsize=(3 * 3, 3 * 3), nrows=3, ncols=3, sharex=True, sharey=True
            )
            clrs = [
                self.COLOR["mid"],
                self.COLOR["mid"],
                self.COLOR["dark_highlight"],
                self.COLOR["dark_highlight"],
                self.COLOR["superfine"],
                self.COLOR["superfine"],
                "green",
                "green",
                self.COLOR["light"],
            ]
            lbls = [
                "$u_x$",
                "$u_y$",
                "$u_x^{out}-u_x^{in}$",
                "$u_y^{out}-u_y^{in}$",
                "$\partial_x p$",
                "$\partial_y p$",
                "$f_x$",
                "$f_y$",
                r"$\nabla \cdot u$",
            ]
            axes = axs.reshape(-1)
        elif self.use_difu and not self.use_difp:
            fig, axs = plt.subplots(
                figsize=(3 * 3, 3 * 3), nrows=3, ncols=3, sharex=True, sharey=True
            )
            clrs = [
                self.COLOR["mid"],
                self.COLOR["mid"],
                self.COLOR["dark_highlight"],
                self.COLOR["dark_highlight"],
                self.COLOR["superfine"],
                "green",
                "green",
            ]
            lbls = [
                "$u_x$",
                "$u_y$",
                "$u_x^{out}-u_x^{in}$",
                "$u_y^{out}-u_y^{in}$",
                "$f_x$",
                "$f_y$",
                r"$\nabla \cdot u$",
            ]
            fig.delaxes(axs.reshape(-1)[-1])
            fig.delaxes(axs.reshape(-1)[-2])
            axes = axs.reshape(-1)[:-2]
        elif self.use_difu and self.use_diff:
            fig, axs = plt.subplots(
                figsize=(3 * 3, 3 * 4), nrows=4, ncols=3, sharex=True, sharey=True
            )
            clrs = [
                self.COLOR["mid"],
                self.COLOR["mid"],
                self.COLOR["dark_highlight"],
                self.COLOR["dark_highlight"],
                "green",
                "green",
                self.COLOR["dark_highlight"],
                self.COLOR["dark_highlight"],
                self.COLOR["light"],
                self.COLOR["dark"],
            ]
            lbls = [
                "$u_x$",
                "$u_y$",
                "$u_x^{out}-u_x^{in}$",
                "$u_y^{out}-u_y^{in}$",
                "$f_x$",
                "$f_y$",
                "$f_x^{out}-f_x^{in}$",
                "$f_y^{out}-f_y^{in}$",
                r"$\nabla \cdot u$",
                "$p^{out}-p^{in}$",
            ]
            fig.delaxes(axs.reshape(-1)[-1])
            fig.delaxes(axs.reshape(-1)[-2])
            axes = axs.reshape(-1)[:-2]
        elif self.use_difu and self.without_f:
            fig, axs = plt.subplots(
                figsize=(3 * 3, 3 * 2), nrows=2, ncols=3, sharex=True, sharey=True
            )
            clrs = [
                self.COLOR["mid"],
                self.COLOR["mid"],
                self.COLOR["dark_highlight"],
                self.COLOR["dark_highlight"],
                self.COLOR["light"],
                self.COLOR["dark"],
            ]
            lbls = [
                "$u_x$",
                "$u_y$",
                "$u_x^{out}-u_x^{in}$",
                "$u_y^{out}-u_y^{in}$",
                r"$\nabla \cdot u$",
                "$p^{out}-p^{in}$",
            ]
            axes = axs.reshape(-1)
        elif self.use_difu:
            fig, axs = plt.subplots(
                figsize=(3 * 3, 3 * 3), nrows=3, ncols=3, sharex=True, sharey=True
            )
            clrs = [
                self.COLOR["mid"],
                self.COLOR["mid"],
                self.COLOR["dark_highlight"],
                self.COLOR["dark_highlight"],
                "green",
                "green",
                self.COLOR["light"],
                self.COLOR["dark"],
            ]
            lbls = [
                "$u_x$",
                "$u_y$",
                "$u_x^{out}-u_x^{in}$",
                "$u_y^{out}-u_y^{in}$",
                "$f_x$",
                "$f_y$",
                r"$\nabla \cdot u$",
                "$p^{out}-p^{in}$",
            ]
            fig.delaxes(axs.reshape(-1)[-1])
            axes = axs.reshape(-1)[:-1]
        elif self.use_gradp_training and not self.use_difu:
            fig, axs = plt.subplots(
                figsize=(4 * 3, 4 * 2), nrows=2, ncols=4, sharey=True
            )
            fig.delaxes(axs.reshape(-1)[-1])
            clrs = [
                self.COLOR["mid"],
                self.COLOR["mid"],
                self.COLOR["superfine"],
                self.COLOR["superfine"],
                "green",
                "green",
                self.COLOR["light"],
            ]
            lbls = ["ux", "uy", "px", "py", "fx", "fy", "div"]
            axes = axs.reshape(-1)[:-1]
        else:
            fig, axs = plt.subplots(
                figsize=(3 * 3, 3 * 2), nrows=2, ncols=3, sharey=True
            )
            clrs = [
                COLOR["mid"],
                COLOR["mid"],
                COLOR["superfine"],
                "green",
                "green",
                COLOR["light"],
            ]
            lbls = ["ux", "uy", "$p$", "fx", "fy", "div"]
            axes = axs.reshape(-1)
        # x = [0, 0, self.L, self.L, 0]
        # y = [0., 1., 1., 0., 0.]
        xs = np.linspace(self.x_start, self.x_end, 100)
        y_top = self.calc_y_top(xs)
        y_bottom = self.calc_y_bottom(xs)
        ms = 3
        for i, ax in enumerate(axes):
            ax.plot(xs, y_top, color="k")
            ax.plot(xs, y_bottom, color="k")
            # if lbls[i] == '$p$' and self.use_difp == True:
            #     ax.plot(self.r[-1][:, 0], self.r[-1][:, 1], ls='None', marker='o',
            #             markersize=5, color=COLOR['dark'], label='$p^{out}-p^{in}$', alpha=0.5)
            #     ax.legend(fontsize='small')
            ax.plot(
                self.r[i][:, 0],
                self.r[i][:, 1],
                ls="None",
                marker="o",
                markersize=ms,
                color=clrs[i],
            )
            if self.__class__.__name__ == "SinusoidalCylinder":
                if i == 0:
                    num_surface = 100
                    r_surface = self.make_r_surface(num_surface)
                ax.plot(r_surface[:, 0], r_surface[:, 1], color="k")
            ax.set_title(lbls[i])
            ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        if show:
            plt.show()
        if save:
            dir_path = f"{path}/fig"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            fig.savefig(f"{dir_path}/train.png")
        plt.clf()
        plt.close()

    # def generate_training_data(self, u_num, p_num, f_num, f_pad, div_num, div_pad):
    #     """
    #     Args
    #     u_num  :number of u training poitns at each boundary
    #     p_num  :number of p training poitns at each boundary
    #     f_num  :list of number of f training points [h,w]
    #     f_pad  : distance from boundary for f
    #     div_num:list of number of div training points [h,w]
    #     div_pad:distance from boundary for div
    #     """
    #     self.r = []
    #     self.f = []
    #     self.generate_u(u_num)
    #     self.generate_f(f_num, f_pad)
    #     self.generate_div(div_num, div_pad)
    #     return self.r, self.f

    def generate_difu(self, difu_num, difu_loc="inlet_outlet"):
        # difu_loc = 'all_inside'
        if difu_loc == "inlet_outlet":
            # use difu at inlet and outlet
            r_difu = self.make_r_mesh_sinusoidal(
                self.x_start,
                self.x_end,
                -self.w / 2 + self.slide,
                self.w / 2 - self.slide,
                1,
                difu_num,
                self.slide,
            )
        elif difu_loc == "all_inside":
            difu_num_x, difu_num_y = self.num_to_num_x_y(difu_num)
            r_difu = self.make_r_mesh_sinusoidal(
                self.x_start - self.L / 2,
                self.x_end - self.L / 2,
                -self.w / 2 + self.slide,
                self.w / 2 - self.slide,
                difu_num_x,
                difu_num_y,
                self.slide,
            )
        difu = np.full(len(r_difu), 0.0)
        # for difux and difuy, use same points
        self.r += [r_difu, r_difu]
        self.f += [difu, difu]

    def generate_diff(self, diff_num):
        r_diff = self.make_r_mesh_sinusoidal(
            self.x_start,
            self.x_end,
            -self.w / 2 + self.slide,
            self.w / 2 - self.slide,
            1,
            diff_num,
            self.slide,
        )
        diff = np.full(len(r_diff), 0.0)
        # for difux and difuy, use same points
        self.r += [r_diff, r_diff]
        self.f += [diff, diff]
