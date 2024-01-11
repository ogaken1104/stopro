import numpy as np
from jax.config import config

from stopro.data_generator.sin_1D import Sin1D
from stopro.data_generator.sin_1D_naive import Sin1DNaive
from stopro.data_preparer.data_preparer import DataPreparer


def prepare(
    project_name,
    simulation_name,
    use_existing_params=False,
    system_name="sin_1D",
    ly_num=10,
):
    data_preparer = DataPreparer(
        # project_name, simulation_name, class_data_generator=Sin1DNaive
        project_name,
        simulation_name,
        class_data_generator=Sin1D,
    )
    data_preparer.create_directory()
    data_preparer.load_params(
        system_name=system_name, use_existing_params=use_existing_params
    )
    #### Here, modify the parameters manually ###
    # data_preparer.params_setting["use_pbc_points"] = True
    # data_preparer.params_generate_test
    # data_preparer.params_generate_training["y_loc"] = "origin"
    data_preparer.params_generate_training["ly_num"] = ly_num
    # data_preparer.params_kernel_arg
    # data_preparer.params_num_points
    # data_preparer.params_plot
    params_optimization = data_preparer.params_main["optimization"]
    params_optimization["maxiter_GD"] = 1600
    params_model = data_preparer.params_main["model"]
    data_preparer.params_main["optimization"]["maxiter_scipy"] = [0]

    data_preparer.params_main["optimization"]["print_process"] = False
    #############################################
    data_preparer.update_params()
    data_preparer.make_data(
        plot_training=False, plot_test=False, save_data=True, save_plot=False
    )
    data_preparer.save_params_prepare()
    data_preparer.save_params_main()
    data_preparer.save_lbls()


def test_sin_1D_prepare():
    config.update("jax_enable_x64", True)
    project_name = "test"
    # y_num_range = [5, 10, 20, 40, 80, 160, 320, 640]
    prepare(project_name, f"sin_1D_direct", use_existing_params=False, ly_num=10)
