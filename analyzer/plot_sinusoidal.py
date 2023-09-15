import cmocean as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from stopro.data_generator.sinusoidal import Sinusoidal
from stopro.data_handler.data_handle_module import HdfOperator
import stopro.analyzer.make_each_plot as plot_modules


def plot_each_sinusoidal(contents, params, lbls, vnames, infer_p):
    hdf_operator = HdfOperator()

    plot_param = params["plot"]
    val_limits = plot_param["val_limits"]
    std_limit = plot_param["std_limit"]
    error_limit = plot_param["error_limit"]
    x_for_ux = plot_param["x_for_ux"]

    abs_error, abs_error, rel_error = hdf_operator.load_analysis_data(
        ["abs_error", "abs_error", "rel_error"], vnames["analysis"]
    )
    f, std = hdf_operator.load_infer_data(lbls["infer"], vnames["infer"])
    r_test, f_test = hdf_operator.load_test_data(lbls["test"], vnames["test"])
    r_train, f_train = hdf_operator.load_train_data(lbls["train"], vnames["train"])

    sinusoidal = Sinusoidal(0.2, 2.5, 1)
    xs = np.linspace(0, sinusoidal.L, 100)
    y_top = sinusoidal.calc_y_top(xs)
    y_bottom = sinusoidal.calc_y_bottom(xs)

    # prepare grid for pcolormesh
    test_num = params["test_data"]["test_num"]
    maximum_height = sinusoidal.w / 2 + sinusoidal.a
    ############ analytical solution #####################
    y_num = test_num
    #####################################################
    ############### use spm result ####################
    y_num = 18
    ######################################
    ################ use fem result ###############
    y_num = test_num
    ##############################################
    x_num = len(r_test[0][::y_num])
    y_grid = np.linspace(-maximum_height, maximum_height, y_num)
    x_grid = np.linspace(0, sinusoidal.L, x_num)
    if "all" in contents:
        # plot for output_all
        if infer_p:
            fig, axes = plt.subplots(figsize=(5 * 4, 4 * 3), nrows=3, ncols=3)
        else:
            fig, axes = plt.subplots(figsize=(5 * 3, 4 * 2), nrows=2, ncols=3)
        for vname, r_te, err, f_infer, st, axs, vlimit in zip(
            vnames["test"], r_test, abs_error, f, std, axes, val_limits
        ):
            cmaps = [cmo.cm.dense, cmo.cm.balance, cmo.cm.dense]
            # 値のプロット
            dot_size = 25
            index = vnames["test"].index(vname)
            ax_index = 0
            axs[ax_index].set_title(vname)
            f[index] = np.array(f[index])
            f_for_plot = sinusoidal.change_outside_values_to_zero(
                r_test[index], f[index]
            )
            f_mesh = f_for_plot.reshape(y_num, x_num)
            mappable = axs[ax_index].pcolormesh(
                x_grid,
                y_grid,
                f_mesh,
                cmap=cmaps[index],
                vmin=vlimit[0],
                vmax=vlimit[1],
                shading="nearest",
            )
            # mappable = axs[ax_index].scatter(r_test[index][:, 0], r_test[index][:, 1], c=f[index], marker='o',
            #                                  s=dot_size, cmap=cmaps[index], vmin=vlimit[0], vmax=vlimit[1])
            divider = make_axes_locatable(axs[ax_index])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(mappable, cax=cax)
            axs[ax_index].set_aspect("equal", adjustable="box")

            # 誤差のプロット
            ax_index = 1
            axs[ax_index].set_title("error")
            abs_error[index] = np.array(abs_error[index])
            f_for_plot = sinusoidal.change_outside_values_to_zero(
                r_test[index], abs_error[index]
            )
            f_mesh = f_for_plot.reshape(y_num, x_num)
            mappable = axs[ax_index].pcolormesh(
                x_grid,
                y_grid,
                f_mesh,
                cmap=mpl.cm.cool,
                norm=mpl.colors.LogNorm(vmin=error_limit[0], vmax=error_limit[1]),
                shading="nearest",
            )
            # mappable = axs[ax_index].scatter(r_test[index][:, 0], r_test[index][:, 1], c=abs_error[index], marker='o',
            #                                  s=dot_size, cmap=mpl.cm.cool, norm=mpl.colors.LogNorm(vmin=error_limit[0], vmax=error_limit[1]))
            divider = make_axes_locatable(axs[ax_index])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(mappable, cax=cax)
            axs[ax_index].set_aspect("equal", adjustable="box")

            for ax in axs:
                ax.plot(xs, y_top, color="k")
                ax.plot(xs, y_bottom, color="k")
            # 標準偏差のプロット
            ax_index = 2
            axs[ax_index].set_title("std")
            std[index] = np.array(std[index])
            f_for_plot = sinusoidal.change_outside_values_to_zero(
                r_test[index], std[index]
            )
            f_mesh = f_for_plot.reshape(y_num, x_num)
            mappable = axs[ax_index].pcolormesh(
                x_grid,
                y_grid,
                f_mesh,
                cmap=mpl.cm.cool,
                norm=mpl.colors.LogNorm(vmin=std_limit[0], vmax=std_limit[1]),
                shading="nearest",
            )
            # mappable = axs[ax_index].scatter(r_test[index][:, 0], r_test[index][:, 1], c=std[index], marker='o',
            #                                  s=dot_size, cmap=mpl.cm.cool, vmin=std_limit[0], vmax=std_limit[1])
            divider = make_axes_locatable(axs[ax_index])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(mappable, cax=cax)
            axs[ax_index].set_aspect("equal", adjustable="box")
        # if sinusoidal.__class__.__name__ == "SinusoidalCylinder":
        #     for i, ax in enumerate(axes.reshape(-1)):
        #         if i == 0:
        #             num_surface = 100
        #             r_surface = sinusoidal.make_r_surface(num_surface)
        #         ax.plot(r_surface[:, 0], r_surface[:, 1], color="k")
        fig.tight_layout()
        fig.savefig(
            f"../fig/output_all.png",
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()

    if "rel_error" in contents:
        fig, axs = plt.subplots(figsize=(5 * 3, 4), ncols=2)
        for vname, r_te, f_te, rel_err, ax in zip(
            vnames["test"], r_test, f_test, rel_error, axs
        ):
            ax.plot(xs, y_top, color="k")
            ax.plot(xs, y_bottom, color="k")
            ax.set_title(vname)
            dot_size = 25
            true_max = np.max(f_te)
            zero_threshold = true_max * 1e-7
            r_te_plot = r_te[np.where(abs(f_te) > zero_threshold)]
            # print(np.where(f_te != 0.))
            mappable = ax.scatter(
                r_te_plot[:, 0],
                r_te_plot[:, 1],
                c=rel_err,
                marker="o",
                s=dot_size,
                cmap=mpl.cm.cool,
                norm=mpl.colors.LogNorm(vmin=0.001, vmax=1.0),
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(mappable, cax=cax)
            ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(
            f"../fig/rel_error.png",
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
        clr = plot_modules.COLOR["dark"]
        lbl = "loss"
        # ax.plot(range(5,100,5),np.array(loss),color=clr,label=lbl)
        ax.plot(loss_index, loss, color=clr)
        ax.set_ylabel("loss", fontsize=22)
        ax.set_xlabel("iteration", fontsize=22)
        fig.savefig(f"../fig/loss.png")
        plt.clf()
        plt.close()

    if "norm_of_grads" in contents:
        norm_of_grads = hdf_operator.load_record("norm_of_grads")
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        clr = plot_modules.COLOR["dark"]
        ax.plot(np.arange(1, len(norm_of_grads) + 1), norm_of_grads, color=clr)
        ax.set_ylabel("norm of gradients", fontsize=22)
        ax.set_xlabel("iteration", fontsize=22)
        fig.savefig(f"../fig/norm_of_gradients.png")
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
            cmaps = [cmo.cm.delta, cmo.cm.delta]
            # 値のプロット
            dot_size = 25
            index = vnames["test"].index(vname)
            ax_index = 0
            axs[ax_index].set_title(vname)
            mappable = axs[ax_index].scatter(
                r_test[index][:, 0],
                r_test[index][:, 1],
                c=f[index],
                marker="o",
                s=dot_size,
                cmap=cmaps[index],
                vmin=vlimit[0],
                vmax=vlimit[1],
            )
            divider = make_axes_locatable(axs[ax_index])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(mappable, cax=cax)
            axs[ax_index].set_aspect("equal", adjustable="box")

            # 誤差のプロット
            ax_index = 1
            axs[ax_index].set_title("error")
            mappable = axs[ax_index].scatter(
                r_test[index][:, 0],
                r_test[index][:, 1],
                c=abs_error[index],
                marker="o",
                s=dot_size,
                cmap=mpl.cm.cool,
                norm=mpl.colors.LogNorm(vmin=error_limit[0], vmax=error_limit[1]),
            )
            divider = make_axes_locatable(axs[ax_index])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(mappable, cax=cax)
            axs[ax_index].set_aspect("equal", adjustable="box")

            for ax in axs:
                ax.plot(xs, y_top, color="k")
                ax.plot(xs, y_bottom, color="k")
            # 標準偏差のプロット
            ax_index = 2
            axs[ax_index].set_title("std")
            mappable = axs[ax_index].scatter(
                r_test[index][:, 0],
                r_test[index][:, 1],
                c=std[index],
                marker="o",
                s=dot_size,
                cmap=mpl.cm.cool,
                vmin=std_limit[0],
                vmax=std_limit[1],
            )
            divider = make_axes_locatable(axs[ax_index])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(mappable, cax=cax)
            axs[ax_index].set_aspect("equal", adjustable="box")

        fig.tight_layout()
        fig.savefig(f"../fig/gradp_all.png", bbox_inches="tight")
        plt.clf()
        plt.close()