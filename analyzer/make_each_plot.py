# import cmocean as cmo
import cmocean as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


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


def create_mesh(arr, num_edge):
    arr_mesh = arr.reshape(num_edge, num_edge)
    return arr_mesh


def plot_val_lattice(p, vname, ax, vlimit, grid):
    vname_for_title_dict = {"ux": "u_x", "uy": "u_y", "px": "p_x", "py": "p_y"}
    try:
        vname_for_title = vname_for_title_dict[vname]
    except:
        vname_for_title = vname
    ax.set_title(rf"${vname_for_title}$")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    if vname == "ux":
        cmap = cmo.cm.dense
    elif vname == "uy":
        cmap = cmo.cm.balance
    elif vname == "p":
        cmap = cmo.cm.amp
    else:
        cmap = cmo.cm.delta
    mappable = ax.pcolormesh(
        grid, grid, p, cmap=cmap, vmin=vlimit[0], vmax=vlimit[1], shading="nearest"
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    ax.set_aspect("equal", adjustable="box")
    return mappable, cax


def plot_error_lattice(err, vname, ax, limit, grid):
    ax.set_title("error")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    mappable = ax.pcolormesh(
        grid,
        grid,
        err,
        cmap=mpl.cm.cool,
        norm=mpl.colors.LogNorm(vmin=limit[0], vmax=limit[1]),
        shading="nearest",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    ax.set_aspect("equal", adjustable="box")
    return mappable, cax


def plot_std_lattice(p, vname, ax, limit, grid):
    ax.set_title("std")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    mappable = ax.pcolormesh(
        grid,
        grid,
        p,
        cmap=mpl.cm.cool,
        shading="nearest",
        norm=mpl.colors.LogNorm(vmin=limit[0], vmax=limit[1]),
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    ax.set_aspect("equal", adjustable="box")
    return mappable, cax


def plot_loss(hdf_operator):
    # plot for loss vs iteration
    itr_step = 1
    loss = hdf_operator.load_record("fun")
    loss = loss[1:]  # Nelder-Meadによる最適化分を除外してプロット
    loss_index = np.arange(len(loss)) * itr_step
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    clr = COLOR["dark"]
    # ax.plot(range(5,100,5),np.array(loss),color=clr,label=lbl)
    ax.plot(loss_index, loss, color=clr, marker="o")
    ax.set_ylabel("loss", fontsize=22)
    ax.set_xlabel("iteration", fontsize=22)
    fig.tight_layout()
    fig.savefig(f"../fig/loss.png")
    plt.clf()
    plt.close()


def plot_each_lattice(contents, plot_param, lbls, vnames):
    # val_limits = [[0., 0.12], [-0.01, 0.01], [0., 1.]]
    # error_limit = [0.0001, 0.1]
    # std_limit = [0, 0.002]
    # x_for_ux = 0.5
    hdf_operator = HdfOperator()

    # # plotに必要なパラメータの読み込み
    # with open(f'{data_path}/param_plot.yaml', 'r') as file:
    #     param_plot = yaml.safe_load(file)
    #     val_limits = param_plot['val_limits']
    #     std_limit = param_plot['std_limit']
    #     error_limit = param_plot['error_limit']
    #     x_for_ux = param_plot['x_for_ux']

    val_limits = plot_param["val_limits"]
    std_limit = plot_param["std_limit"]
    error_limit = plot_param["error_limit"]
    x_for_ux = plot_param["x_for_ux"]

    abs_error = hdf_operator.load_analysis_data(["abs_error"], ["ux", "uy", "p"])[0]
    f, std = hdf_operator.load_infer_data(["f", "std"], ["ux", "uy", "p"])
    r_test, f_test = hdf_operator.load_test_data(["r", "f"], ["ux", "uy", "p"])

    if "all" in contents:
        # plot for output_all
        fig, axes = plt.subplots(figsize=(5 * 3, 4 * 3), nrows=3, ncols=3)
        for vname, r_te, err, f_infer, st, axs, vlimit in zip(
            vnames["test"], r_test, abs_error, f, std, axes, val_limits
        ):
            num_edge = int(np.sqrt(len(r_te)))
            grid = np.linspace(0, 1, num_edge)
            err_mesh = create_mesh(err, num_edge)
            f_infer_mesh = create_mesh(f_infer, num_edge)
            std_mesh = create_mesh(st, num_edge)

            shrink = 1
            # 値のプロット
            mappable, cax0 = plot_val_lattice(f_infer_mesh, vname, axs[0], vlimit, grid)
            fig.colorbar(mappable, cax=cax0)

            # 誤差のプロット
            mappable, cax1 = plot_error_lattice(
                err_mesh, vname, axs[1], error_limit, grid
            )
            fig.colorbar(mappable, cax=cax1)

            # 標準偏差のプロット
            mappable, cax2 = plot_std_lattice(std_mesh, vname, axs[2], std_limit, grid)
            fig.colorbar(mappable, cax=cax2)

        fig.tight_layout()
        fig.savefig(
            f"../fig/output_all.png",
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()

    if "ux" in contents:
        # plot the values of ux
        x_for_ux_range = np.linspace(0, 1, 5)
        for x_for_ux in x_for_ux_range:
            r_ux = r_test[0]
            plot_index = r_ux[:, 0] == x_for_ux
            r_plot = r_ux[plot_index][:, 1]
            f_test_plot = f_test[0][plot_index]
            f_infer_plot = f[0][plot_index]
            fig, ax = plt.subplots()
            ax.plot(
                r_plot,
                f_test_plot,
                color=COLOR["superfine"],
                label="analytical",
                linestyle="--",
            )
            ax.plot(r_plot, f_infer_plot, color=COLOR["mid"], label="inference")
            ax.set_title(f"ux at x = {x_for_ux}")
            ax.legend()
            fig.savefig(
                f"../fig/ux_{x_for_ux}.png",
                bbox_inches="tight",
            )
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


def plot_each_gradp(contents, plot_param, lbls, vnames):
    hdf_operator = HdfOperator()

    val_limits = plot_param["val_limits"]
    std_limit = plot_param["std_limit"]
    error_limit = plot_param["error_limit"]
    x_for_ux = plot_param["x_for_ux"]

    abs_error = hdf_operator.load_analysis_data(
        ["abs_error"], ["ux", "uy", "px", "py"]
    )[0]
    f, std = hdf_operator.load_infer_data(["f", "std"], ["ux", "uy", "px", "py"])
    r_test, f_test = hdf_operator.load_test_data(["r", "f"], ["ux", "uy", "px", "py"])

    if "all" in contents:
        # plot for output_all
        fig, axes = plt.subplots(figsize=(5 * 3, 4 * 3), nrows=2, ncols=3)
        for vname, r_te, err, f_infer, st, axs, vlimit in zip(
            vnames["test"], r_test, abs_error, f, std, axes, val_limits
        ):
            num_edge = int(np.sqrt(len(r_te)))
            grid = np.linspace(0, 1, num_edge)
            err_mesh = create_mesh(err, num_edge)
            f_infer_mesh = create_mesh(f_infer, num_edge)
            std_mesh = create_mesh(st, num_edge)

            shrink = 1
            # 値のプロット
            mappable, cax0 = plot_val_lattice(f_infer_mesh, vname, axs[0], vlimit, grid)
            fig.colorbar(mappable, cax=cax0)

            # 誤差のプロット
            mappable, cax1 = plot_error_lattice(
                err_mesh, vname, axs[1], error_limit, grid
            )
            fig.colorbar(mappable, cax=cax1)

            # 標準偏差のプロット
            mappable, cax2 = plot_std_lattice(std_mesh, vname, axs[2], std_limit, grid)
            fig.colorbar(mappable, cax=cax2)

        fig.tight_layout()
        fig.savefig(f"../fig/output_all.png", bbox_inches="tight")
        plt.clf()
        plt.close()

    if "ux" in contents:
        # plot the values of ux
        x_for_ux_range = np.linspace(0, 1, 5)
        for x_for_ux in x_for_ux_range:
            r_ux = r_test[0]
            plot_index = r_ux[:, 0] == x_for_ux
            r_plot = r_ux[plot_index][:, 1]
            f_test_plot = f_test[0][plot_index]
            f_infer_plot = f[0][plot_index]
            fig, ax = plt.subplots()
            ax.plot(
                r_plot,
                f_test_plot,
                color=COLOR["superfine"],
                label="analytical",
                linestyle="--",
            )
            ax.plot(r_plot, f_infer_plot, color=COLOR["mid"], label="inference")
            ax.set_title(f"ux at x = {x_for_ux}")
            ax.legend()
            fig.savefig(
                f"../fig/ux_{x_for_ux}.png",
                bbox_inches="tight",
            )
            plt.clf()
            plt.close()

    if "gradp" in contents:
        fig, axes = plt.subplots(figsize=(5 * 3, 4 * 3), nrows=2, ncols=3)
        val_limits = [[-1.0, 1.0], [-1.0, 1.0]]
        for vname, r_te, err, f_infer, st, axs, vlimit in zip(
            vnames["test"][2:],
            r_test[2:],
            abs_error[2:],
            f[2:],
            std[2:],
            axes,
            val_limits,
        ):
            print(vname)
            num_edge = int(np.sqrt(len(r_te)))
            grid = np.linspace(0, 1, num_edge)
            err_mesh = create_mesh(err, num_edge)
            f_infer_mesh = create_mesh(f_infer, num_edge)
            std_mesh = create_mesh(st, num_edge)

            shrink = 1
            # 値のプロット
            mappable, cax0 = plot_val_lattice(f_infer_mesh, vname, axs[0], vlimit, grid)
            fig.colorbar(mappable, cax=cax0)

            # 誤差のプロット
            mappable, cax1 = plot_error_lattice(
                err_mesh, vname, axs[1], error_limit, grid
            )
            fig.colorbar(mappable, cax=cax1)

            # 標準偏差のプロット
            mappable, cax2 = plot_std_lattice(std_mesh, vname, axs[2], std_limit, grid)
            fig.colorbar(mappable, cax=cax2)

        fig.tight_layout()
        fig.savefig(f"../fig/gradp_all.png", bbox_inches="tight")
        plt.clf()
        plt.close()

    if "ux_wall" in contents:
        fig, ax = plt.subplots()
        x_ans = [0.0, 1.0]
        ux_ans = [1.0, 1.0]
        ax.plot(x_ans, ux_ans, color=COLOR["superfine"], label="answer", linestyle="--")
        r_ux = r_test[0]
        index_for_plot = r_ux[:, 1] == 1.0
        r_ux_for_plot = r_ux[index_for_plot][:, 0]
        ux_for_plot = f_infer[0][index_for_plot]
        std_for_plot = std[0][index_for_plot]
        ax.plot(r_ux_for_plot, ux_for_plot, color=COLOR["mid"], label="inference")
        ax.fill_between(
            r_ux_for_plot,
            ux_for_plot - 2 * std_for_plot,
            ux_for_plot + 2 * std_for_plot,
            color=COLOR["dark"],
            alpha=0.3,
            label=r"$\pm 2\sigma$",
        )
        fig.savefig(f"../fig/ux_upper_wall.png", bbox_inches="tight")
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
