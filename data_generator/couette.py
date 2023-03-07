import numpy as np

from stopro.data_generator.poiseuille import Poiseuille


class Couette(Poiseuille):
    def __init__(self, Re=None, L=None, ux_upper_wall=None, slide=0.03, random_arrange=True, use_gradp_training=False, infer_gradp=False, infer_ux_wall=False, use_only_bottom_u=False, use_only_inlet_gradp=False, cut_last_x=False) -> None:
        super().__init__(Re, L, None, None, slide,
                         random_arrange, use_gradp_training, infer_gradp, infer_ux_wall, use_only_bottom_u, use_only_inlet_gradp, cut_last_x)
        self.ux_upper_wall = ux_upper_wall
        self.f_ux = lambda r: ux_upper_wall * r[:, 1]
        self.f_p = lambda r: np.zeros(len(r))
        self.gradpx = 0.
        self.infer_ux_wall = infer_ux_wall

    def generate_u(self, u_num, u_b=0, ux_num_inside=0):
        if self.infer_ux_wall:
            np.random.seed(42)
            r_ux_bottom_wall = self.make_r_mesh(
                0., self.L, 0., 0., int(u_num/2), 1)
            if ux_num_inside:
                r_ux_inside = np.random.uniform(
                    0.+self.slide, self.L-self.slide, (100, 2))
                r_ux_inside = r_ux_inside[:ux_num_inside]
                # r_ux = np.concatenate([r_ux_inside, r_ux_bottom_wall])
                r_ux = r_ux_inside
                r_uy = r_ux_inside
            else:
                r_ux = r_ux_bottom_wall
            ux = self.f_ux(r_ux)

            # r_uy = self.make_r_mesh(0., self.L, 0., 0., int(u_num/2), 1)
            uy = np.zeros(len(r_uy))

            self.r += [r_ux, r_uy]
            self.f += [ux, uy]
        else:
            return super().generate_u(u_num, u_b)

    def generate_test(self, num, ux_test=False):
        r = self.make_r_mesh(0., self.L, 0., 1., num, num)
        if self.infer_ux_wall:
            if ux_test:
                r_ux = self.make_r_mesh(0., self.L, 1., 1., num*10, 1)
            else:
                r_ux = r
            ux = self.f_ux(r_ux)
            r_uy = r
            uy = np.zeros(len(r_uy))

            self.r_test += [r_ux, r_uy]
            self.f_test += [ux, uy]

            return self.r_test, self.f_test
        else:
            return super().generate_test(num, ux_test)

    def generate_training_data(self, u_num, p_num, f_num, f_pad, div_num, div_pad, ux_num_inside=0):
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
        self.generate_u(u_num, ux_num_inside=ux_num_inside)
        if self.use_gradp_training:
            self.generate_gradp(p_num)
        else:
            self.generate_p(p_num)
        self.generate_f(f_num, f_pad)
        self.generate_div(div_num, div_pad)
        return self.r, self.f
