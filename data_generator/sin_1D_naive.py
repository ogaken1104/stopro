import matplotlib.pyplot as plt
import numpy as np

from stopro.data_generator.data_generator import DataGenerator


class Sin1DNaive(DataGenerator):
    def __init__(self, use_pbc_points=False, seed: int = 0, num_period: float = 1.0):
        self.x_min = 0.0
        self.x_max = np.pi * 2 * num_period
        self.r = []
        self.f = []
        self.r_test = []
        self.f_test = []
        self.use_pbc_points = use_pbc_points
        self.seed = seed

    def generate_y(self, y_num):
        r_y = np.linspace(self.x_min, self.x_max, y_num)
        y = np.sin(r_y)
        self.r += [r_y]
        self.f += [y]

    def generate_test(self, test_num, test_side: bool = True):
        if test_side:
            r = np.linspace(-np.pi / 2, 2.5 * np.pi, test_num)
        else:
            r = np.linspace(0.0, self.x_max, test_num)
        y = np.sin(r)
        self.r_test += [r]
        self.f_test += [y]
        return self.r_test, self.f_test

    def generate_pbc_y(self):
        r_pbc = np.array([self.x_min])
        f_pbc = np.array([0.0])
        self.r += [r_pbc]
        self.f += [f_pbc]

    def generate_training_data(self, y_num, sigma2_noise: float = None):
        self.generate_y(y_num)
        if sigma2_noise:
            self.f[0] = self.add_white_noise(self.f[0], sigma2_noise)
        if self.use_pbc_points:
            self.generate_pbc_y()
        return self.r, self.f

    def plot_train(self, save=False, path=None, show=False):
        fig, ax = plt.subplots()
        ax.plot(
            self.r_test[0], self.f_test[0], label="answer", linestyle="--", color="k"
        )
        ax.scatter(
            self.r[0],
            self.f[0],
            label=r"$y$ training",
            marker="o",
            color="k",
            facecolor="None",
        )
        if self.use_pbc_points:
            ax.scatter(self.r[1], self.f[1], label="pbc", marker="+", color="k", s=140)
            index_ly = 2
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
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
