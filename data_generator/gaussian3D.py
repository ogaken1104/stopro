import matplotlib.pyplot as plt
import numpy as np

from stopro.data_generator.data_generator import DataGenerator
from mpl_toolkits.mplot3d import Axes3D


class Gaussian3D(DataGenerator):
    def __init__(self, seed: int = 0):
        self.x_min = -1.0
        self.x_max = 1.0
        self.r = []
        self.f = []
        self.r_test = []
        self.f_test = []
        self.seed = seed
        self.gaussian = lambda r: np.exp(np.sum(-(r**2), axis=1))

    def make_r_mesh(
        self, x_start, x_end, y_start, y_end, z_start, z_end, numx, numy, numz
    ):
        x = np.linspace(x_start, x_end, numx)
        y = np.linspace(y_start, y_end, numy)
        z = np.linspace(z_start, z_end, numz)
        xx, yy, zz = np.meshgrid(x, y, z)
        r = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=1)
        return r

    def generate_f(self, f_num):
        r = self.make_r_mesh(
            self.x_min,
            self.x_max,
            self.x_min,
            self.x_max,
            self.x_min,
            self.x_max,
            f_num,
            f_num,
            f_num,
        )
        f = self.gaussian(r)
        self.r += [r]
        self.f += [f]

    def generate_test(self, test_num, z_plane=0.1):
        if z_plane is not None:
            r = self.make_r_mesh(
                self.x_min,
                self.x_max,
                self.x_min,
                self.x_max,
                z_plane,
                z_plane,
                test_num,
                test_num,
                1,
            )
        f = self.gaussian(r)
        self.r_test += [r]
        self.f_test += [f]
        return self.r_test, self.f_test

    def generate_training_data(self, f_num, sigma2_noise: float = None):
        self.generate_f(f_num)
        if sigma2_noise:
            self.f[0] = self.add_white_noise(self.f[0], sigma2_noise)
        return self.r, self.f

    def plot_train(self, save=False, path=None, show=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")  # 3Dサブプロットを追加

        ax.scatter(self.r[0][:, 0], self.r[0][:, 1], self.r[0][:, 2], c="k")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.legend()
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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")  # 3Dサブプロットを追加

        ax.scatter(
            self.r_test[0][:, 0], self.r_test[0][:, 1], self.r_test[0][:, 2], c="k"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
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
