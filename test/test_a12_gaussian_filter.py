from stopro.sub_modules.gaussian_filter import GaussianFilter
from scipy.optimize import minimize

import numpy as np


def test_gaussian_filter():
    start = 0
    end = np.pi * 2
    num_basis = 90
    num_pred = 100
    length = 0.05
    maxiter = 2
    r_basis = np.linspace(start, end, num_basis)
    f_basis = np.sin(r_basis)
    r_pred = np.linspace(start, end, num_pred)
    gaussian_filter = GaussianFilter(length)

    # def loss(length):
    #     gaussian_filter.length = length
    #     f_basis_inf = gaussian_filter.interpolate(r_basis, r_basis, f_basis)
    #     return np.sum((f_basis_inf - f_basis) ** 2)

    # opt = {"maxiter": maxiter, "disp": 0}
    # res = minimize(loss, length, options=opt)
    # length_optimized = res["x"]
    length_optimized = length
    print(length_optimized)

    gaussian_filter.length = length_optimized
    f_pred = gaussian_filter.interpolate(r_pred, r_basis, f_basis)
    assert np.allclose(f_pred, np.sin(r_pred), atol=1e-3)
