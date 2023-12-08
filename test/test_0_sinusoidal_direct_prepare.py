from jax.config import config

from stopro.data_generator.sinusoidal import Sinusoidal
from stopro.data_preparer.data_preparer import DataPreparer


def prepare(
    project_name,
    simulation_name,
    use_existing_params=False,
    system_name="sinusoidal_new",
):
    data_preparer = DataPreparer(
        project_name, simulation_name, class_data_generator=Sinusoidal
    )
    data_preparer.create_directory()
    data_preparer.load_params(
        system_name=system_name, use_existing_params=use_existing_params
    )
    data_preparer.params_main["model"]["init_kernel_hyperparameter"] = {
        "uxux": [0.0, -1.0, -1.0],
        "uyuy": [0.0, -1.0, -1.0],
        "pp": [0.0, -1.0, -1.0],
        # "uxuy": [0.0, 0.0, 0.0]
        # "noise": float(np.log(1.0e-06)),
    }
    data_preparer.params_main["optimization"]["maxiter_GD"] = 5000  # 5000
    data_preparer.params_main["optimization"]["lr"] = 1.0e-02  # 1e-02
    data_preparer.params_main["optimization"]["maxiter_scipy"] = [0]
    data_preparer.params_main["optimization"]["eps"] = 0.0001
    data_preparer.params_main["optimization"]["interval_check"] = 300
    #############################################
    data_preparer.update_params()
    data_preparer.params_kernel_arg = ["uxux", "uyuy", "pp"]

    data_preparer.make_data(plot_training=False, plot_test=False, save=True)
    data_preparer.save_params_prepare()
    data_preparer.save_params_main()
    data_preparer.save_lbls()


def test_sinusoidal_direct_prepare():
    config.update("jax_enable_x64", True)
    project_name = "test"
    prepare(project_name, f"sinusoidal_direct")
