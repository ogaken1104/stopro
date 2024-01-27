import numpy as np
from jax.config import config

from stopro.data_generator.drag3D import Drag3D
from stopro.data_preparer.data_preparer import DataPreparer


def prepare(
    project_name,
    simulation_name,
    use_existing_params=False,
    system_name="drag3D",
):
    data_preparer = DataPreparer(
        # project_name, simulation_name, class_data_generator=Sin1DNaive
        project_name,
        simulation_name,
        class_data_generator=Drag3D,
    )
    data_preparer.create_directory()
    data_preparer.load_params(
        system_name=system_name, use_existing_params=use_existing_params
    )
    #### Here, modify the parameters manually ###
    data_preparer.params_setting["particle_radius"] = 0.4
    data_preparer.params_generate_test["test_num"] = 40
    # data_preparer.params_generate_test
    # data_preparer.params_kernel_arg
    # data_preparer.params_num_points
    # data_preparer.params_plot
    #############################################
    data_preparer.update_params()
    data_preparer.make_data(
        plot_training=True, plot_test=False, save_data=True, save_plot=True
    )
    data_preparer.save_params_prepare()
    data_preparer.save_params_main()
    data_preparer.save_lbls()


def test_sin_1D_prepare():
    config.update("jax_enable_x64", True)
    project_name = "test"
    # y_num_range = [5, 10, 20, 40, 80, 160, 320, 640]
    prepare(project_name, f"drag3D", use_existing_params=False)
