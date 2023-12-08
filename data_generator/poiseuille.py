import os
import pickle

import cmocean as cmo
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from stopro.data_generator.stokes_2D_generator import StokesDataGenerator

# adimentionalize so that H,ρ,η=1.0


class Poiseuille(StokesDataGenerator):
    def __init__(
        self,
        Re=None,
        L=None,
        delta_p=None,
        pin=None,
        slide=0.03,
        random_arrange=True,
        use_gradp_training=False,
        infer_gradp=False,
        infer_ux_wall=False,
        use_only_bottom_u=False,
        use_only_inlet_gradp=False,
        cut_last_x=False,
        use_difp=False,
        use_difu=False,
    ):
        super().__init__(random_arrange)
        self.Re = Re
        self.L = L
        self.delta_p = delta_p
        self.pin = pin
        self.slide = slide
        if self.delta_p is not None:
            self.f_ux = (
                lambda r: -0.5 * self.delta_p * self.Re * r[:, 1] * (1 - r[:, 1])
            )
            self.f_duxdy = lambda r: -0.5 * self.delta_p * self.Re * (1 - 2 * r[:, 1])
            self.f_p = lambda r: pin + self.delta_p * r[:, 0]
            self.gradpx = self.delta_p / self.L
        #         self.f_uy=lambda r :0.
        #         self.f_p=lambda r:self.delta_p/self.L*r[0]+self.pin
        self.r = []
        self.f = []
        self.r_test = []
        self.f_test = []
        self.use_gradp_training = use_gradp_training
        self.gradpy = 0.0
        self.infer_gradp = infer_gradp
        self.use_only_bottom_u = use_only_bottom_u
        self.use_only_inlet_gradp = use_only_inlet_gradp
        self.cut_last_x = cut_last_x
        self.use_difp = use_difp
        self.infer_ux_wall = infer_ux_wall
        self.use_difu = use_difu

    # generation of training data
    def generate_u(self, u_num, u_b=0.0, u_1D2C=False):
        """
        premise: ux,uy values are taken at same points
        """
        if self.infer_ux_wall:
            np.random.seed(42)
            r_ux_inside = np.random.uniform(
                0.0 + self.slide, self.L - self.slide, (100, 2)
            )
            r_ux_inside = r_ux_inside[:u_num]
            # r_ux = np.concatenate([r_ux_inside, r_ux_bottom_wall])
            r_ux = r_ux_inside
            r_uy = r_ux_inside
        else:
            u_num = int(u_num / 2)
            # define x_last
            if self.cut_last_x:
                x_last = self.L - self.L / u_num
            else:
                x_last = self.L
            # define num_along_y
            if self.use_only_bottom_u:
                num_along_y = 1
            elif u_1D2C:
                num_along_y = 7
            else:
                num_along_y = 2
            if self.random_arrange:
                r_ux = self.make_r_mesh_random(
                    0.0 + self.slide,
                    x_last - self.slide,
                    0.0,
                    1.0,
                    u_num - 2,
                    num_along_y,
                    self.slide,
                    "x",
                )
                r_ux = np.concatenate(
                    [
                        r_ux,
                        np.array(
                            [[0.0, 0.0], [0.0, 1.0], [x_last, 0.0], [x_last, 1.0]]
                        ),
                    ]
                )
                r_uy = self.make_r_mesh_random(
                    0.0 + self.slide,
                    x_last - self.slide,
                    0.0,
                    1.0,
                    u_num - 2,
                    num_along_y,
                    self.slide,
                    "x",
                )
                r_uy = np.concatenate(
                    [
                        r_uy,
                        np.array(
                            [[0.0, 0.0], [0.0, 1.0], [x_last, 0.0], [x_last, 1.0]]
                        ),
                    ]
                )
            else:
                r_ux = self.make_r_mesh(0.0, x_last, 0.0, 1.0, u_num, num_along_y)
                r_uy = self.make_r_mesh(0.0, x_last, 0.0, 1.0, u_num, num_along_y)
            # without_p test
            # r_ux_1 = self.make_r_mesh(0., 1., 0., 1., u_num, 2)
            # r_uy_1 = self.make_r_mesh(0., 1., 0., 1., u_num, 2)
            # r_ux_2 = self.make_r_mesh(0., 1., 0., 1., 2, u_num)
            # r_uy_2 = self.make_r_mesh(0., 1., 0., 1., 2, u_num)
            # r_ux = np.concatenate([r_ux_1, r_ux_2])
            # r_uy = np.concatenate([r_uy_1, r_uy_2])
            # r_ux = np.unique(r_ux, axis=0)
            # r_ux = np.unique(r_uy, axis=0)
        # ux = -0.5*self.delta_p*self.Re*r_ux[:, 1]*(1-r_ux[:, 1])
        ux = self.f_ux(r_ux)  # 本当にこれでいけるかチェック
        uy = np.zeros(len(r_uy))
        self.r += [r_ux, r_uy]
        self.f += [ux, uy]

    def generate_test(self, test_num, ux_test=False, infer_duxdy_boundary=False):
        if infer_duxdy_boundary:
            num_x = 100
            r = self.make_r_mesh(0.0, self.L, 0.0, 1.0, num_x, 2)
            duxdy = self.f_duxdy(r)
            self.r_test = [r]
            self.f_test = [duxdy]
            return self.r_test, self.f_test
        if not ux_test:
            r = self.make_r_mesh(0.0, self.L, 0.0, 1.0, test_num, test_num)
            ux_test = self.f_ux(r)
            uy_test = np.zeros(len(r))
            r_ux = r
            r_uy = r
            if not self.use_gradp_training:
                p_test = self.f_p(r)
                r_p = r
            elif self.infer_gradp:
                px_test = np.full(len(r), self.gradpx)
                py_test = np.full(len(r), self.gradpy)
        elif ux_test:
            r_ux = self.make_r_mesh(0.0, self.L, 0.0, 1.0, 5, 100)
            ux_test = self.f_ux(r_ux)
            r_uy = np.array([[0.0, 0.0]])
            uy_test = np.zeros(len(r_uy))
            r_p = np.array([[0.0, 0.0]])
            if not self.use_gradp_training:
                p_test = self.f_p(r_p)

        if self.use_gradp_training and not self.infer_gradp:
            self.r_test = [r_ux, r_uy]
            self.f_test = [ux_test, uy_test]
        elif self.use_gradp_training and self.infer_gradp:
            self.r_test = [r_ux, r_uy, r, r]
            self.f_test = [ux_test, uy_test, px_test, py_test]
        else:
            self.r_test = [r_ux, r_uy, r_p]
            self.f_test = [ux_test, uy_test, p_test]

        return self.r_test, self.f_test

    def generate_p(self, p_num):
        #         pout=self.pin+self.delta_p*self.L
        #         x_p0=np.full(p_num,0)
        #         x_p1=np.full(p_num,self.L)
        #         y_p=np.linspace(0.,1.,p_num)
        #         r_p=np.stack([np.concatenate([x_p0,x_p1]),np.concatenate([y_p,y_p])],axis=1)
        #         p0=np.full(p_num,self.pin)
        #         p1=np.full(p_num,pout)
        #         p=np.concatenate([p0,p1])
        p_num = int(p_num / 2)
        # define x_last
        if self.cut_last_x:
            x_last = self.L - self.L / p_num
        else:
            x_last = self.L
        if self.use_only_inlet_gradp:
            num_along_y = 1
        else:
            num_along_y = 2

        if self.random_arrange:
            r_p = self.make_r_mesh_random(
                0.0,
                x_last,
                0.0 + self.slide,
                1.0 - self.slide,
                num_along_y,
                p_num - 2,
                self.slide,
                "y",
            )
            r_p = np.concatenate(
                [r_p, np.array([[x_last, 0.0], [x_last, 1.0], [0.0, 0.0], [0.0, 1.0]])]
            )
        elif not self.use_difp:
            r_p = self.make_r_mesh(0.0, x_last, 0.0, 1.0, num_along_y, p_num)
            # without_p test
            # r_p = np.array([[0., 0.]])
        elif self.use_difp:
            r_p = np.array([[0.0, 0.5]])

        p = self.f_p(r_p)
        self.r += [r_p]
        self.f += [p]

    def generate_gradp(self, p_num):
        p_num = int(p_num / 2)
        if self.use_only_inlet_gradp:
            num_along_x = 1
        else:
            num_along_x = 2
        if self.random_arrange:
            r_p = self.make_r_mesh_random(
                0.0,
                1.0,
                0.0 + self.slide,
                1.0 - self.slide,
                num_along_x,
                p_num - 2,
                self.slide,
                "y",
            )
            r_p = np.concatenate(
                [r_p, np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0]])]
            )
        else:
            r_p = self.make_r_mesh(
                0.0, 1.0, 0.0 + self.slide, 1.0 - self.slide, num_along_x, p_num
            )
            # without_p test
            # r_p = np.array([[0., 0.]])
        px = np.full(len(r_p), self.gradpx)
        py = np.full(len(r_p), self.gradpy)
        self.r += [r_p, r_p]
        self.f += [px, py]

    def generate_f(self, f_num, fx_pad, force=0.0):
        """
        premise: ux,uy values are taken at same points
        """
        x_start = 0.0 + fx_pad
        x_end = self.L - fx_pad
        y_start = 0.0 + fx_pad
        y_end = 1.0 - fx_pad
        if self.random_arrange:
            f_num_gen = int(np.sqrt(f_num) * 1.3)
            r = r = self.make_r_mesh(
                x_start, x_end, y_start, y_end, f_num_gen, f_num_gen
            )
            r_fx = (np.random.random_sample(r.shape) - 0.5) * self.slide + r
            r_fy = (np.random.random_sample(r.shape) - 0.5) * self.slide + r
            #         fx=np.full(f_num,force)
            total_fx = np.arange(len(r_fx))
            total_fy = np.arange(len(r_fy))
            np.random.shuffle(total_fx)
            np.random.shuffle(total_fy)
            r_fx = r_fx[total_fx[:f_num]]
            r_fy = r_fy[total_fy[:f_num]]
        else:
            r_fx = self.make_r_mesh(x_start, x_end, y_start, y_end, f_num, f_num)
            r_fy = self.make_r_mesh(x_start, x_end, y_start, y_end, f_num, f_num)
        fx = np.full(len(r_fx), force)
        fy = np.full(len(r_fy), force)
        self.r += [r_fx, r_fy]
        self.f += [fx, fy]

    def generate_div(self, div_num, div_pad, divu=0.0):
        x_start = 0.0 + div_pad
        x_end = self.L - div_pad
        y_start = 0.0 + div_pad
        y_end = 1.0 - div_pad
        if self.random_arrange:
            div_num_gen = int(np.sqrt(div_num) * 1.3)
            r = self.make_r_mesh(
                x_start, x_end, y_start, y_end, div_num_gen, div_num_gen
            )
            r_div = (np.random.random_sample(r.shape) - 0.5) * self.slide + r
            total_div = np.arange(len(r_div))
            np.random.shuffle(total_div)
            r_div = r_div[total_div[:div_num]]
        else:
            r_div = self.make_r_mesh(x_start, x_end, y_start, y_end, div_num, div_num)
        div = np.full(len(r_div), divu)
        self.r += [r_div]
        self.f += [div]

    def plot_test(self, val_limits, save=False, path=None, show=False):
        fig, axs = plt.subplots(figsize=(5 * 3, 3), ncols=3, sharey=True)
        lbls = ["ux", "uy", "p"]
        x = [0, 0, self.L, self.L, 0]
        y = [0.0, 1.0, 1.0, 0.0, 0.0]
        # axes = axs.reshape(-1)
        for i, ax in enumerate(axs):
            ax.plot(x, y, color="black")
            mappable = ax.scatter(
                self.r_test[i][:, 0],
                self.r_test[i][:, 1],
                c=self.f_test[i],
                marker="o",
                cmap=cmo.cm.dense,
                vmin=val_limits[i][0],
                vmax=val_limits[i][1],
            )
            ax.set_title(lbls[i])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(mappable, cax)
            ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        if save:
            dir_path = f"{path}/fig"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            fig.savefig(f"{dir_path}/test.png")
        if show:
            plt.show()
        plt.clf()
        plt.close()

    #     def generate_test_data(self,u_num_test):
    #         x_ux=np.full(u_num_test,self.L/2.)
    #         y_ux=np.linspace(-1.,1.,u_num_test)
    #         r_ux=np.stack([x_ux,y_ux],axis=1)
    #         return [r_ux]*3

    def plot_train(self, save=False, path=None, show=False):
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
        elif self.use_gradp_training and not self.use_difu:
            fig, axs = plt.subplots(
                figsize=(3 * 3, 3 * 2), nrows=2, ncols=4, sharex=True, sharey=True
            )
            clrs = [
                self.COLOR["mid"],
                self.COLOR["mid"],
                self.COLOR["superfine"],
                self.COLOR["superfine"],
                "green",
                "green",
                self.COLOR["light"],
            ]
            lbls = [
                "$u_x$",
                "$u_y$",
                "$\partial_x p$",
                "$\partial_y p$",
                "fx",
                "fy",
                "div",
            ]
            axes = axs.reshape(-1)[:-1]
        elif len(self.r) == 6:
            fig, axs = plt.subplots(
                figsize=(3 * 3, 3 * 2), nrows=2, ncols=3, sharex=True, sharey=True
            )
            clrs = [
                self.COLOR["mid"],
                self.COLOR["mid"],
                self.COLOR["superfine"],
                "green",
                "green",
                self.COLOR["light"],
            ]
            lbls = ["ux", "uy", "p", "fx", "fy", "div"]
            axes = axs.reshape(-1)
        elif len(self.r) == 5:
            fig, axs = plt.subplots(
                figsize=(3 * 3, 3 * 2), nrows=2, ncols=3, sharex=True, sharey=True
            )
            clrs = [
                self.COLOR["mid"],
                self.COLOR["mid"],
                "green",
                "green",
                self.COLOR["light"],
            ]
            lbls = ["ux", "uy", "fx", "fy", "div"]
            fig.delaxes(axs.reshape(-1)[-1])
            axes = axs.reshape(-1)[:-1]
        else:
            raise ValueError("plot function is not implemented for this case")
        x = [0, 0, self.L, self.L, 0]
        y = [0.0, 1.0, 1.0, 0.0, 0.0]
        for i, ax in enumerate(axes):
            # if i >= len(lbls):
            #     break
            ax.plot(x, y, color="black")
            ax.plot(
                self.r[i][:, 0], self.r[i][:, 1], ls="None", marker="o", color=clrs[i]
            )
            if lbls[i] == "p" and self.use_difp == True:
                ax.plot(
                    self.r[-1][:, 0],
                    self.r[-1][:, 1],
                    ls="None",
                    marker="o",
                    color=self.COLOR["dark"],
                    label="Delta_p",
                )
                ax.legend(fontsize="small")
            ax.set_title(lbls[i])
            ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        if save:
            dir_path = f"{path}/fig"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            fig.savefig(f"{dir_path}/train.png")
        if show:
            plt.show()
        plt.clf()
        plt.close()

    def generate_difp(self, difp_num, difp_pad, difp_loc):
        if difp_loc == "all_inside":
            x_start = 0.0 + difp_pad
            x_end = self.L / 2 - difp_pad
            y_start = 0.0 + difp_pad
            y_end = 1.0 - difp_pad
            r_difp = self.make_r_mesh(
                x_start, x_end, y_start, y_end, difp_num, difp_num, difp_pad
            )
        elif difp_loc == "inlet_outlet":
            # use difp on inlet and outlet
            r_difp = self.make_r_mesh(
                0.0, 0.0, 0.0 + self.slide, 1.0 - self.slide, 1, difp_num
            )
        difp = np.full(len(r_difp), self.delta_p)
        self.r += [r_difp]
        self.f += [difp]

    def generate_difu(self, difu_num):
        # use difu at inlet and outlet
        r_difu = self.make_r_mesh(
            0.0, 0.0, 0.0 + self.slide, 1.0 - self.slide, 1, difu_num
        )
        difu = np.full(len(r_difu), 0.0)
        # for difux and difuy, use same points
        self.r += [r_difu, r_difu]
        self.f += [difu, difu]

    def generate_training_data(
        self,
        u_num=None,
        p_num=None,
        f_num=None,
        f_pad=None,
        div_num=None,
        div_pad=None,
        difp_num=None,
        difp_pad=None,
        difp_loc=None,
        difu_num=None,
        diff_num=None,
        u_1D2C=False,
    ):
        """
        Args
        u_num  :number of u training poitns at each boundary
        p_num  :number of p training poitns at each boundary
        f_num  :list of number of f training points [h,w]
        f_pad  : distance from boundary for f
        div_num:list of number of div training points [h,w]
        div_pad:distance from boundary for div
        """
        self.r = []
        self.f = []
        self.generate_u(u_num, u_1D2C=u_1D2C)
        if self.use_difu:
            self.generate_difu(difu_num)
        if self.use_gradp_training:
            self.generate_gradp(p_num)
        elif p_num:
            self.generate_p(p_num)
        self.generate_f(f_num, f_pad)
        if self.use_diff:
            self.generate_diff(diff_num)
        self.generate_div(div_num, div_pad)
        if self.use_difp:
            self.generate_difp(difp_num, difp_pad, difp_loc)
        return self.r, self.f
