import matplotlib.pyplot as plt
import numpy as np

from stopro.data_generator.data_generator import DataGenerator


class Sin1D(DataGenerator):
    def __init__(self, use_pbc_points=False, seed: int = 0):
        self.x_min = 0.0
        self.x_max = np.pi * 2
        self.r = []
        self.f = []
        self.r_test = []
        self.f_test = []
        self.use_pbc_points = use_pbc_points
        self.seed = seed

    def generate_y(self, y_loc):
        if y_loc == "both":
            r_y = np.array([self.x_min, self.x_max])
        elif y_loc == "origin":
            r_y = np.array([0.0])
        y = np.sin(r_y)
        self.r += [r_y]
        self.f += [y]

    def generate_ly(self, ly_num):
        # r_ly = np.linspace(self.x_min, self.x_max, ly_num)
        r_ly = np.linspace(self.x_min, self.x_max, ly_num + 1)[:-1]
        # 同じ点を2点与えても発散しないかテスト（error termを加えているから発散しないはず）
        # r_ly = np.concatenate([r_ly, np.array([0., 0.])])
        ly = -np.sin(r_ly)
        self.r += [r_ly]
        self.f += [ly]

    def generate_test(self, test_num):
        r = np.linspace(-np.pi, 3 * np.pi, test_num)
        y = np.sin(r)
        self.r_test += [r]
        self.f_test += [y]
        return self.r_test, self.f_test

    def generate_pbc_y(self):
        r_pbc = np.array([self.x_min])
        f_pbc = np.array([0.0])
        self.r += [r_pbc]
        self.f += [f_pbc]

    def generate_training_data(self, ly_num, y_loc="both", sigma2_noise: float = None):
        self.generate_y(y_loc)
        if sigma2_noise:
            self.f[0] = self.add_white_noise(self.f[0], sigma2_noise)
        if self.use_pbc_points:
            self.generate_pbc_y()
        self.generate_ly(ly_num)
        return self.r, self.f

    def plot_train(self, save=False, path=None, show=False):
        fig, ax = plt.subplots()
        ax.plot(
            self.r_test[0], self.f_test[0], label="answer", linestyle="--", color="k"
        )
        ax.scatter(
            self.r[0],
            np.zeros(len(self.r[0])),
            label=r"$y$ training",
            marker="o",
            color="k",
            facecolor="None",
        )
        index_ly = 1
        if self.use_pbc_points:
            ax.scatter(self.r[1], self.f[1], label="pbc", marker="+", color="k", s=140)
            index_ly = 2
        ax.scatter(
            self.r[index_ly],
            np.zeros(len(self.r[index_ly])),
            label=r"$\Delta y$ training",
            marker="x",
            color="k",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.0))
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
