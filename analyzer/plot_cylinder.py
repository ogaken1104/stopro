import cmocean as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import stopro.analyzer.make_each_plot as plot_modules
from stopro.data_generator.cylinder import Cylinder
from stopro.data_generator.drag import Drag
from stopro.data_handler.data_handle_module import HdfOperator


def plot_each_cylinder(contents, params, lbls, vnames):
    hdf_operator = HdfOperator()

    plot_param = params['plot']
    val_limits = plot_param['val_limits']
    std_limit = plot_param['std_limit']
    error_limit = plot_param['error_limit']

    abs_error, abs_error, rel_error = hdf_operator.load_analysis_data(
        ['abs_error', 'abs_error', 'rel_error'], vnames['analysis'])
    f, std = hdf_operator.load_infer_data(lbls['infer'], vnames['infer'])
    r_test, f_test = hdf_operator.load_test_data(lbls['test'], vnames['test'])
    r_train, f_train = hdf_operator.load_train_data(
        lbls['train'], vnames['train'])

    sample = Cylinder(**params['training_data']['condition'])

    r_ux = r_test[0]
    num_per_side = int(np.sqrt(len(r_ux)))
    x_grid = np.linspace(0., np.max(r_ux[:, 0]), num_per_side)
    y_grid = np.linspace(0., np.max(r_ux[:, 1]), num_per_side)

    if 'all' in contents:
        plot_output_all(vnames, f, abs_error, std, num_per_side,
                        x_grid, y_grid, val_limits, error_limit, std_limit, sample)
    if 'loss' in contents:
        plot_modules.plot_loss(hdf_operator)
    if 'rel_error' in contents:
        plot_rel_error(r_test, f_test, rel_error)


def plot_output_all(vnames, f, abs_error, std, num_per_side, x_grid, y_grid, val_limits, error_limit, std_limit, sample):
    fig, axes = plt.subplots(figsize=(5*3, 5*2), nrows=2, ncols=3,
                             sharex=True, sharey=True)
    cmaps_val = [cmo.cm.dense, cmo.cm.balance]
    shading = "gouraud"
    for i, axs in enumerate(axes):
        # plot values
        ax_index = 0
        # axs[ax_index].set_title(vnames[i])
        f[i] = np.array(f[i])
        f_mesh = f[i].reshape(num_per_side, num_per_side)
        sample.plot_heatmap(fig, axs[ax_index], x_grid, y_grid, f_mesh, shading=shading,
                            vmin=val_limits[i][0], vmax=val_limits[i][1], cmap=cmaps_val[i])

        # plot abs error
        ax_index = 1
        # axs[ax_index].set_title()
        abs_error[i] = np.array(abs_error[i])
        f_mesh = abs_error[i].reshape(num_per_side, num_per_side)
        sample.plot_heatmap(fig, axs[ax_index], x_grid, y_grid, f_mesh, shading=shading,
                            vmin=error_limit[0], vmax=error_limit[1], cmap=mpl.cm.cool, norm=mpl.colors.LogNorm)

        # plot std
        ax_index = 2
        std[i] = np.array(std[i])
        f_mesh = std[i].reshape(num_per_side, num_per_side)
        sample.plot_heatmap(fig, axs[ax_index], x_grid, y_grid, f_mesh, shading=shading,
                            vmin=std_limit[0], vmax=std_limit[1], cmap=mpl.cm.cool, norm=mpl.colors.LogNorm)
    fig.tight_layout()
    fig.savefig(f'../fig/output_all.png',
                bbox_inches="tight", )
    plt.clf()
    plt.close()


def plot_rel_error(r_test, f_test, rel_error):
    fig, axs = plt.subplots(figsize=(5*2, 5), ncols=2, sharey=True)
    for i, ax in enumerate(axs):
        r_te = r_test[i]
        f_te = f_test[i]
        true_max = np.max(f_te)
        zero_threshold = true_max*1e-07
        r_te_plot = r_te[np.where(abs(f_te) > zero_threshold)]
        mappable = ax.scatter(r_te_plot[:, 0], r_te_plot[:, 1], c=rel_error[i],
                              s=10, cmap=mpl.cm.cool, norm=mpl.colors.LogNorm(vmin=0.001, vmax=1.))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(mappable, cax=cax)
        ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(f'../fig/rel_error.png',
                bbox_inches="tight", )
    plt.clf()
    plt.close()
