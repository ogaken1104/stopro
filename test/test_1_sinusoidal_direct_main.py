import argparse
import time
from functools import partial
import copy

import jax.numpy as jnp
import yaml
from jax import grad, jacfwd, jacrev, jit
from jax.config import config

from stopro.analyzer.analysis import analyze_result, analyze_error_interval
from stopro.analyzer.make_each_plot import *
from stopro.analyzer.plot_sin_1D_each import plot_each_sin1D
from stopro.analyzer.plot_sinusoidal import plot_each_sinusoidal
from stopro.data_handler.data_handle_module import *
from stopro.GP.gp_sinusoidal_without_p import GPSinusoidalWithoutP
from stopro.GP.gp_sinusoidal_independent import GPSinusoidalWithoutPIndependent
from stopro.GP.gp_sinusoidal_4kernels import GPSinusoidal4Kernels
from stopro.GP.gp_sinusoidal_infer_difp import GPSinusoidalInferDifP

from stopro.GP.kernels import define_kernel
from stopro.solver.optimizers import optimize_by_adam
from stopro.sub_modules.init_modules import get_init, reshape_init
from stopro.sub_modules.load_modules import load_params, load_data
from stopro.sub_modules.loss_modules import hessian, logposterior


config.update("jax_enable_x64", True)


def test_sinusoidal_direct_main():
    optimize = 1
    data_path = "test/sinusoidal_direct"

    # load params
    params_main, params_prepare, lbls = load_params(
        params_path=f"{data_path}/data_input"
    )
    params_model = params_main["model"]
    params_optimization = params_main["optimization"]
    params_plot = params_prepare["plot"]
    vnames = params_prepare["vnames"]
    params_setting = params_prepare["setting"]
    params_generate_training = params_prepare["generate_training"]
    params_generate_test = params_prepare["generate_test"]
    params_kernel_arg = params_prepare["kernel_arg"]

    # prepare initial hyper-parameter
    init = get_init(
        params_model["init_kernel_hyperparameter"],
        params_model["kernel_type"],
        system_type=params_model["system_type"],
    )
    if params_model["kernel_type"] == "sm":
        init = reshape_init(init, params_kernel_arg)

    # prepare data
    hdf_operator = HdfOperator(data_path)
    r_test, μ_test, r_train, μ_train, f_train = load_data(lbls, vnames, hdf_operator)
    delta_y_train = jnp.empty(0)
    for i in range(len(r_train)):
        delta_y_train = jnp.append(delta_y_train, f_train[i] - μ_train[i])
    del f_train
    del μ_train
    print(delta_y_train)

    args_predict = r_test, μ_test, r_train, delta_y_train, params_model["epsilon"]

    # setup model
    Kernel = define_kernel(params_model)
    gp_model = GPSinusoidalWithoutPIndependent(
        use_difp=params_setting["use_difp"],
        use_difu=params_setting["use_difu"],
        lbox=jnp.array([2.5, 0.0]),
        infer_governing_eqs=params_prepare["generate_test"]["infer_governing_eqs"],
        Kernel=Kernel,
        index_optimize_noise=params_model["index_optimize_noise"],
    )
    gp_model.set_constants(*args_predict)
    loglikelihood, predictor = (
        gp_model.trainingFunction_all,
        gp_model.predictingFunction_all,
    )
    func = jit(logposterior(loglikelihood, params_optimization))
    dfunc = jit(grad(func, 0))
    hess = hessian(func)

    ############## training ####################
    # compiletion for training
    theta_for_compile = init.at[0].set(
        init[0] + jnp.array(0.01)
    )  # slide theta a little bit
    maxiter_GD = copy.deepcopy(params_optimization["maxiter_GD"])
    params_optimization["maxiter_GD"] = 1
    optimize_by_adam(
        func, dfunc, hess, theta_for_compile, params_optimization, *args_predict[2:]
    )
    params_optimization["maxiter_GD"] = maxiter_GD

    ############## training ####################
    train_start_time = time.time()
    # optimize hyperparameters
    if optimize:
        opts = [{"x": init}]
        opt, loss, theta, norm_of_grads_list = optimize_by_adam(
            func, dfunc, hess, init, params_optimization, *args_predict[2:]
        )
    else:
        opt = init
        n_training = sum([r.shape[0] for r in r_train])
        loss = [func(init, *args_predict[2:]) / n_training]
        theta = [init]
        norm_of_grads_list = []
    train_end_time = time.time()
    print(f"{train_end_time - train_start_time:.1f}s: training")

    _, f_test = hdf_operator.load_test_data(lbls["test"], vnames["test"])
    ############ check abs error @ designated iteration #########
    interval_check = params_optimization["interval_check"]
    if interval_check:
        num_check = int((len(loss) - 1) / interval_check)
        index_check = np.arange(0, num_check * interval_check + 1, interval_check)
        mean_abs_error_interval = [[], []]
        f_infer_interval = [[], []]
        for i in index_check:
            theta_check = theta[i]
            fs, Σs = predictor(theta_check, *args_predict)
            f_infer = [f for f in fs]
            ux_error, uy_error = analyze_error_interval(f_test, f_infer)
            mean_abs_error_interval[0].append(ux_error)
            mean_abs_error_interval[1].append(uy_error)
            f_infer_interval[0].append(f_infer[0])
            f_infer_interval[1].append(f_infer[1])
        hdf_operator.save_analysis_data(
            ["mean_abs_error_interval"], vnames["analysis"], [mean_abs_error_interval]
        )
        hdf_operator.save_analysis_data(
            ["infer_interval"], vnames["analysis"], [f_infer_interval]
        )

    ############ prediction ################
    # completion for predict
    fs, Σs = predictor(theta_for_compile, *args_predict)
    ###### actual inference ######
    pred_start_time = time.time()
    fs, Σs = predictor(opt, *args_predict)
    pred_end_time = time.time()
    print(f"{pred_end_time - pred_start_time:.1f}s: prediction")

    # unpack values
    f_infer = [f for f in fs]
    std = [jnp.sqrt(jnp.diag(Σ)) for Σ in Σs]

    # save results
    hdf_operator.save_record(lbls["record"], [theta, loss, norm_of_grads_list])
    hdf_operator.save_infer_data(lbls["infer"], vnames["infer"], [f_infer, std])

    # save progress
    run_time = train_end_time - train_start_time + pred_end_time - pred_start_time
    print(f"completed in {run_time:.1f} (sec)\n")

    # analysis
    vals_list_analysis = analyze_result(
        f_test,
        f_infer,
        std,
        theta,
        loss,
        norm_of_grads_list,
        params_kernel_arg,
        vnames["analysis"],
        analysis_text_path=f"test/sinusoidal_direct/data_output/analysis.txt",
    )
    hdf_operator.save_analysis_data(
        lbls["analysis"], vnames["analysis"], vals_list_analysis
    )

    mean_rel_error = vals_list_analysis[5]

    assert np.all(np.array(mean_rel_error) < 0.1)
