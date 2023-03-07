import os

import matplotlib.pyplot as plt
import numpy as np

from stopro.data_generator.data_generator import DataGenerator


class StokesDataGenerator(DataGenerator):
    def __init__(self, random_arrange=True):
        self.random_arrange = random_arrange
        COLORS = ["#DCBCBC", "#C79999", "#B97C7C", "#A25050", "#8F2727",
                  "#7C0000",     "#DCBCBC20", "#8F272720", "#00000060"]
        self.COLOR = {i[0]: i[1] for i in zip(['light', 'light_highlight', 'mid',  'mid_highlight',
                                               'dark', 'dark_highlight', 'light_trans', 'dark_trans', 'superfine'], COLORS)}
        pass

    def make_r_mesh(self, x_start, x_end, y_start, y_end, numx, numy):
        x = np.linspace(x_start, x_end, numx)
        y = np.linspace(y_start, y_end, numy)
        xx, yy = np.meshgrid(x, y)
        r = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
        return r

    def make_r_mesh_random(self, x_start, x_end, y_start, y_end, numx, numy, slide, shuf='all'):
        r = self.make_r_mesh(x_start, x_end, y_start, y_end, numx, numy)
        if shuf == 'all':
            r = r+(np.random.random_sample(r.shape)-0.5)*slide
        elif shuf == 'x':
            r[:, 0] = r[:, 0] + \
                (np.random.random_sample(r[:, 0].shape)-0.5)*slide
        elif shuf == 'y':
            r[:, 1] = r[:, 1] + \
                (np.random.random_sample(r[:, 1].shape)-0.5)*slide
        return self.delete_out(r)

    def delete_out(self, r):
        a = r[np.all(r <= 1., axis=1)]
        b = a[np.all(0. <= a, axis=1)]
        return b

    def generate_training_data(self, u_num=None, p_num=None, f_num=None, f_pad=None, div_num=None, div_pad=None, difp_num=None, difp_pad=None, difp_loc=None, difu_num=None):
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
        self.generate_u(u_num)
        if self.use_difu:
            self.generate_difu(difu_num)
        if self.use_gradp_training:
            self.generate_gradp(p_num)
        elif p_num:
            self.generate_p(p_num)
        self.generate_f(f_num, f_pad)
        self.generate_div(div_num, div_pad)
        if self.use_difp:
            self.generate_difp(difp_num, difp_pad, difp_loc)
        return self.r, self.f

    #     # plot of training data designating index
    # def plot_each_data(self, index):
    #     num = len(index)
    #     fig, ax = plt.subplots(figsize=(4, 2*num), nrows=num, sharex=True)
    #     clrs = [COLOR['mid'], COLOR['mid'], COLOR['superfine'],
    #             'green', 'green', COLOR['light']]
    #     lbls = ['ux', 'uy', 'p', 'fx', 'fy', 'div']
    #     x = [0, 0, self.L, self.L, 0]
    #     y = [-1., 1., 1., -1., -1.]
    #     for i, j in enumerate(index):
    #         ax[i].plot(x, y, color='black')
    #         ax[i].plot(self.r[j][:, 0], self.r[j][:, 1],
    #                    ls='None', marker='o', COLOR=clrs[j])
    #         ax[i].set_title(lbls[j])
    #     fig.tight_layout()
    #     plt.show()

    def plot_train(self, save=False, path=None):

        fig, axs = plt.subplots(
            figsize=(3*3, 3*2), nrows=2, ncols=3, sharex=True, sharey=True)
        clrs = [self.COLOR['mid'], self.COLOR['mid'], self.COLOR['superfine'],
                'green', 'green', self.COLOR['light']]
        lbls = ['ux', 'uy', 'p', 'fx', 'fy', 'div']
        x = [0, 0, self.L, self.L, 0]
        y = [0., 1., 1., 0., 0.]
        axes = axs.reshape(-1)
        for i, ax in enumerate(axes):
            ax.plot(x, y, color='black')
            ax.plot(self.r[i][:, 0], self.r[i][:, 1],
                    ls='None', marker='o', color=clrs[i])
            ax.set_title(lbls[i])
            ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        if save:
            dir_path = f'{path}/fig'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            fig.savefig(f'{dir_path}/train.png')
        plt.clf()
        plt.close()

    def plot_test(self):
        pass

    def plot_result(self):
        pass
