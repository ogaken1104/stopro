import cmocean as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from stopro.data_generator.sinusoidal import Sinusoidal
from stopro.data_handler.data_handle_module import HdfOperator

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


def plot_each_sin1D(contents, params_plot, params_prepare, lbls, vnames):
    hdf_operator = HdfOperator()
    val_limits = params_plot["val_limits"]
    std_limit = params_plot["std_limit"]
    error_limit = params_plot["error_limit"]

    abs_error, abs_error, rel_error = hdf_operator.load_analysis_data(
        ["abs_error", "abs_error", "rel_error"], vnames["analysis"]
    )
    f, std = hdf_operator.load_infer_data(lbls["infer"], vnames["infer"])
    r_test, f_test = hdf_operator.load_test_data(lbls["test"], vnames["test"])
    r_train, f_train = hdf_operator.load_train_data(lbls["train"], vnames["train"])

    if "all" in contents:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(r_test[0], f[0], label="infer", linestyle="-", color=COLOR["dark"])
        # plot test data
        ax.plot(r_test[0], f_test[0], label="answer", linestyle="--", color="k")
        ax.scatter(
            r_train[0],
            f_train[0],
            label=r"$y$ training",
            marker="o",
            color="k",
            facecolor="None",
        )
        index_ly = 1
        if len(r_train) == 3:
            ax.scatter(
                r_train[1], f_train[1], label="pbc", marker="+", color="k", s=140
            )
            index_ly = 2
        if len(r_train) >= 2:
            ax.scatter(
                r_train[index_ly],
                f_train[index_ly],
                label=r"$\Delta y$ train",
                marker="x",
                color="k",
            )
        # ax.plot(r_train[0], y_min*np.ones_like(r_train[0])+shift, ls='None', marker='o',
        #         ms=12, label=r'$y$ training', color='k')
        # ax.plot(r_train[1], y_min*np.ones_like(r_train[1])+shift, ls='None', marker='x',
        #         ms=12, label=r'$\Delta y$ training', color='k')
        ax.fill_between(
            r_test[0],
            f[0] - 2 * std[0],
            f[0] + 2 * std[0],
            color=COLOR["dark"],
            alpha=0.3,
            label=r"$\pm 2\sigma$",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_ylim(-1.1, 1.1)
        ax.legend()
        fig.savefig(
            f"../fig/plot.png",
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()

    if "abs_error" in contents:
        fig, ax = plt.subplots()
        plt.yscale("log")
        ax.plot(
            r_test[0],
            abs_error[0],
            color="k",
        )
        ax.set_ylabel("absolute_error")
        ax.set_xlabel("x")
        y_min = 1e-06
        shift = y_min * 0.1
        ax.set_ylim(y_min, 1e-02)
        ax.plot(
            r_train[0],
            y_min * np.ones_like(r_train[0]) + shift,
            ls="None",
            marker="o",
            ms=12,
            label=r"$y$ training",
            color="k",
        )
        ax.plot(
            r_train[1],
            y_min * np.ones_like(r_train[1]) + shift,
            ls="None",
            marker="x",
            ms=12,
            label=r"$\Delta y$ training",
            color="k",
        )
        ax.legend()
        fig.savefig(f"../fig/abs_error.png", bbox_inches="tight")
        plt.clf()
        plt.close()

    if "loss" in contents:
        # plot for loss vs iteration
        itr_step = 1
        loss = hdf_operator.load_record("fun")
        loss = loss[1:]  # Nelder-Meadによる最適化分を除外してプロット
        loss_index = np.arange(len(loss)) * itr_step
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        clr = COLOR["dark"]
        lbl = "loss"
        # ax.plot(range(5,100,5),np.array(loss),color=clr,label=lbl)
        ax.plot(loss_index, loss, color=clr)
        ax.set_ylabel("loss", fontsize=22)
        ax.set_xlabel("iteration", fontsize=22)
        fig.savefig(f"../fig/loss.png")
        plt.clf()
        plt.close()
