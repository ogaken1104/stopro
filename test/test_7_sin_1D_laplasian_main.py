import time
from pathlib import Path

import jax.numpy as jnp
from jax import grad, jit
from jax.config import config

from stopro.analyzer.analysis import analyze_result
from stopro.analyzer.make_each_plot import *
from stopro.analyzer.plot_sin_1D_each import plot_each_sin1D
from stopro.analyzer.plot_sinusoidal import plot_each_sinusoidal
from stopro.data_handler.data_handle_module import *
from stopro.GP.gp_1D_laplacian import GPmodel1DLaplacian
from stopro.GP.gp_1D_laplacian_pbc import GPmodel1DLaplacianPbc
from stopro.GP.gp_naive import GPmodelNaive
from stopro.GP.kernels import define_kernel
from stopro.solver.optimizers import optimize_by_adam
from stopro.sub_modules.init_modules import get_init, reshape_init
from stopro.sub_modules.load_modules import load_data, load_params
from stopro.sub_modules.loss_modules import hessian, logposterior

# FP32→FP64に変更することで、小さい値でもnanにならないようにする
config.update("jax_enable_x64", True)


def test_sin_1D_direct_main():
    optimize = 1
    data_path = Path("test/sin_1D_direct")

    # load params
    params_main, params_prepare, lbls = load_params(data_path / "data_input")
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

    args_predict = r_test, μ_test, r_train, delta_y_train, params_model["epsilon"]

    # setup model
    Kernel = define_kernel(params_model)
    gp_model = GPmodel1DLaplacian(Kernel=Kernel)
    loglikelihood, predictor = (
        gp_model.trainingFunction_all,
        gp_model.predictingFunction_all,
    )
    gp_model.set_constants(*args_predict)

    func = jit(logposterior(loglikelihood, params_optimization))
    dfunc = jit(grad(func, 0))
    hess = hessian(func)

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
    train_end_time = time.time()
    print(f"{train_end_time - train_start_time:.1f}s: training")
    # predict
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
    time_4 = time.time()
    # print(f"{time_4 - time_3:.1f}s: analyze and save data")

    # save progress
    run_time = train_end_time - train_start_time + pred_end_time - pred_start_time
    print(f"completed in {run_time:.1f} (sec)\n")

    # analysis
    _, f_test = hdf_operator.load_test_data(lbls["test"], vnames["test"])
    vals_list_analysis = analyze_result(
        f_test,
        f_infer,
        std,
        theta,
        loss,
        norm_of_grads_list,
        params_kernel_arg,
        vnames["analysis"],
        analysis_text_path=data_path / "data_output/analysis.txt",
    )
    hdf_operator.save_analysis_data(
        lbls["analysis"], vnames["analysis"], vals_list_analysis
    )

    mean_abs_error = vals_list_analysis[4]

    assert np.all(np.array(mean_abs_error) < 0.3)
