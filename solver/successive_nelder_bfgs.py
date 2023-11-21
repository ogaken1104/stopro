from functools import partial

import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, value_and_grad
from jax.example_libraries import optimizers
from scipy import optimize

# from evosax import EvoParams, SimAnneal

# NO_GRAD_OPTIMIZE_LIST = ["Nelder-Mead", "Powell", "TNC", "BFGS", "L-BFGS-B"]
NO_GRAD_OPTIMIZE_LIST = ["Nelder-Mead", "Powell"]
## if not use autograd

FIRST_GRAD_OPTIMIZE_LIST = ["TNC", "BFGS", "L-BFGS-B", "CG"]  # ?
SECOND_GRAD_OPTIMIZE_LIST = [
    "CG",
]


def optimize_successive_nelder_bfgs(f, df, hf, init, params_optimization, *args):
    """
    Optimize hyper-parameter using scipy.optimize.minimize
    """
    res = [{"x": init}]
    theta = [init]
    loss = []
    loss_before_optimize = func(theta[0], *args)
    print(f"loss before optimize: {loss_before_optimize}")
    loss.append(loss_before_optimize)
    if jnp.isnan(loss_before_optimize):
        raise Exception("初期条件でlossがnanになりました")

    def save_theta_loss(xk):
        theta.append(xk)
        loss.append(func(xk, *args))

    # _:ignore returns
    r_train, *_ = args
    ntraining = 0
    for r in r_train:
        ntraining += r.shape[0]
    # the number of training points
    func = lambda p, *args: f(p, *args) / ntraining
    dfunc = lambda p, *args: df(p, *args) / ntraining
    hess = lambda p, *args: hf(p, *args) / ntraining

    method_scipy = params_optimization["method_scipy"]
    maxiter_scipy = params_optimization["maxiter_scipy"]
    # maxiter_scipy = int(maxiter_scipy[0] / 2)

    opt = {"maxiter": maxiter_scipy[0], "disp": 0}
    ## Neldear-Mead
    res.append(
        optimize.minimize(
            func,
            res[-1]["x"],
            args=args,
            method="Nelder-Mead",
            jac=None,
            hess=None,
            options=opt,
            callback=save_theta_loss,
        )
    )
    print(
        f"loss after second optimize: {loss[-1]}, \n result of first optimize{res[-1]}"
    )
    iter_per_method = 1
    opt = {"maxiter": iter_per_method, "disp": 0}
    for i in range(int(maxiter_scipy[1] / iter_per_method / 2)):
        ## Neldear-Mead
        res.append(
            optimize.minimize(
                func,
                res[-1]["x"],
                args=args,
                method="Nelder-Mead",
                jac=None,
                hess=None,
                options=opt,
                callback=save_theta_loss,
            )
        )
        if res[-1]["fun"] < -0.5:
            print("loss is enough small")
            break
        ## L-BFGS-B
        res.append(
            optimize.minimize(
                func,
                res[-1]["x"],
                args=args,
                method="L-BFGS-B",
                jac=dfunc,
                hess=None,
                options=opt,
                callback=save_theta_loss,
            )
        )
        if res[-1]["success"] == "True" or res[-1]["fun"] < -0.5:
            print("loss is enough small")
            break

    print(
        f"loss after second optimize: {loss[-1]}, \n result of first optimize{res[-1]}"
    )
    return theta[-1], loss, theta, [0]
