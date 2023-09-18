import yaml
import numpy as np


def load_params(params_path="../data_input"):
    with open(f"{params_path}/params_prepare.yaml") as file:
        params_prepare = yaml.safe_load(file)
    with open(f"{params_path}/params_main.yaml") as file:
        params_main = yaml.safe_load(file)
    with open(f"{params_path}/lbls.yaml") as file:
        lbls = yaml.safe_load(file)
    return params_main, params_prepare, lbls


def load_data(lbls, vnames, hdf_operator):
    r_train, f_train = hdf_operator.load_train_data(lbls["train"], vnames["train"])
    r_test, f_test = hdf_operator.load_test_data(lbls["test"], vnames["test"])
    μ_train = [np.zeros_like(f_tr) for f_tr in f_train]
    μ_test = [np.zeros_like(f_te) for f_te in f_test]
    return r_test, μ_test, r_train, μ_train, f_train
