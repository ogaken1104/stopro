import argparse

from jax.config import config
import numpy as np

from stopro.data_generator.poiseuille import Poiseuille
from stopro.data_preparer.data_preparer import DataPreparer


def prepare(
    project_name,
    simulation_name,
    use_existing_params=False,
    system_name="poiseuille",
    u_num=None,
    sigma2_noise=None,
):
    data_preparer = DataPreparer(
        project_name, simulation_name, class_data_generator=Poiseuille
    )
    data_preparer.create_directory()
    data_preparer.load_params(
        system_name=system_name, use_existing_params=use_existing_params
    )
    #### Here, modify the parameters manually ###
    data_preparer.params_main["optimization"]["maxiter_GD"] = 5000  # 5000
    data_preparer.params_main["optimization"]["lr"] = 1.0e-02  # 1e-02
    data_preparer.params_main["optimization"]["maxiter_scipy"] = [0]
    data_preparer.params_main["optimization"]["eps"] = 0.0001
    # # data_preparer.params_main["optimization"]["interval_check"] = 50
    # #############################################
    data_preparer.update_params()
    # data_preparer.params_kernel_arg = ["uxux", "uyuy", "pp"]

    data_preparer.make_data(plot_training=False, plot_test=False, save=True)
    data_preparer.save_params_prepare()
    data_preparer.save_params_main()
    data_preparer.save_lbls()


def test_poiseuille_direct_prepare():
    config.update("jax_enable_x64", True)
    project_name = "test"
    prepare(project_name, f"poiseuille_direct", use_existing_params=False, u_num=None)
    # for i, sigma2_noise in enumerate(noise_range):
    #     prepare(
    #         project_name,
    #         f"noise_{noise_index[i]}",
    #         use_existing_params=False,
    #         u_num=u_num,
    #         sigma2_noise=sigma2_noise,
    #     )
