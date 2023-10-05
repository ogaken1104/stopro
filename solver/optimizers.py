import jax.numpy as jnp
import optax
from jax import jit, value_and_grad
from jax.example_libraries import optimizers
from scipy import optimize
from functools import partial
import numpy as np

# from evosax import EvoParams, SimAnneal

NO_GRAD_OPTIMIZE_LIST = ["Nelder-Mead", "Powell", "TNC", "BFGS", "L-BFGS-B"]
## if not use autograd

FIRST_GRAD_OPTIMIZE_LIST = ["TNC", "BFGS", "L-BFGS-B", "CG"]  # ?
SECOND_GRAD_OPTIMIZE_LIST = [
    "CG",
]


def optimize_with_scipy(func, dfunc, hess, res, theta, loss, params_optimization, args):
    """
    Optimize hyper-parameter using scipy.optimize.minimize

    Args:
        func (callable): loss function
        dfunc (callable): first derivative of loss function
        hess (callable): second derivative of loss function
        res (callable): list for load result
        theta (jnp.array): array of hyper-parameter
        loss (list): list of loss values
        params_optimization (dict)
        args: arguments of the loss function

        jnp.array: optimized hyper-parameter
    Returns:
        list: result of optimization
        list: loss values
    """

    def save_theta_loss(xk):
        theta.append(xk)
        loss.append(func(xk, *args))

    method_scipy = params_optimization["method_scipy"]
    maxiter_scipy = params_optimization["maxiter_scipy"]

    # if params["model"]["kernel_typ"] == "sm":
    #     theta_array = theta.items()

    for method, maxiter in zip(method_scipy, maxiter_scipy):
        print(method, maxiter)
        opt = {"maxiter": maxiter, "disp": 0}

        if method in NO_GRAD_OPTIMIZE_LIST:
            dfunc = None
            hess = None
        elif method in FIRST_GRAD_OPTIMIZE_LIST:
            hess = None
        elif method in SECOND_GRAD_OPTIMIZE_LIST:
            pass
        else:
            raise Exception("scipy optimize method is not in list")
        res.append(
            optimize.minimize(
                func,
                res[-1]["x"],
                args=args,
                method=method,
                jac=dfunc,
                hess=hess,
                options=opt,
                callback=save_theta_loss,
            )
        )
        print(
            f"loss after optimize by {method}: {loss[-1]}, \n result of first optimize{res[-1]}"
        )
    return res, theta, loss


# def optimize_with_evosax(func, dfunc, hess, res, theta, loss, optimize_param, args):
#     strategy = SimAnneal()
#     es_params = strategy.default_params
#     state = strategy.initialize(res[-1]["x"], es_params)

#     # for t in range(num_generations):
#     #     x, state = strategy.ask(res, state, es_params)
#     #     fitness = ...  # Your population evaluation fct
#     #     state = strategy.tell(x, fitness, state, es_params)


def optimize_by_adam(f, df, hf, init, params_optimization, *args):
    """
    Optimize hyper-parameter using scipy.opimize.minimize and/or gradient dcsnet using optax

    Args:
        f (callable): loss function
        df (callable): first derivative of loss function
        hf (callable): second derivative of loss function
        init (jnp.array): array of hyper-parameter
        optimize_param (dict)
        args: arguments of the loss function

    Returns:
        jnp.array: optimized hyper-parameter
        list: loss values
        list: hyper-parameter at each iteration
        list: norm of the gradients
    """
    maxiter_GD = params_optimization["maxiter_GD"]
    lr = params_optimization["lr"]
    eps = params_optimization["eps"]
    maxiter_scipy = params_optimization["maxiter_scipy"]
    method_GD = params_optimization["method_GD"]
    method_scipy = params_optimization["method_scipy"]
    print_process = params_optimization["print_process"]
    loss = []
    theta = [init]
    norm_of_grads_list = []
    # use_sm_kernel = params["model"]["kernel_type"] == "sm"

    def printer(loss, opt_result, init):
        print(f"\t loss = {loss[-1]:12.6e}, niter = {maxiter_GD:5d}")
        print(f"\t\t     θ0 = {jnp.exp(init)}")
        print(f"\t\t      θ = {jnp.exp(opt_result)}")
        print(f"\t\t log(θ) = {opt_result}")

    # _:ignore returns
    r_train, *_ = args
    ntraining = 0
    for r in r_train:
        ntraining += r.shape[0]
    # the number of training points
    func = lambda p, *args: f(p, *args) / ntraining
    dfunc = lambda p, *args: df(p, *args) / ntraining
    hess = lambda p, *args: hf(p, *args) / ntraining

    index_fixed = params_optimization["index_fixed"]
    if index_fixed:
        index_fixed = jnp.array(index_fixed)
        init_fixed = init[index_fixed]
        init = jnp.delete(init, index_fixed)

        def func_sub_theta(init_fixed, init_optimize, *args):
            ## TODO thetaを復元するプログラムを書く
            init = jnp.zeros(len(init_optimize) + len(init_fixed))
            init = init.at[index_fixed].set(init_fixed)
            index_optimize = jnp.setdiff1d(jnp.arange(len(init)), index_fixed)
            init = init.at[index_optimize].set(init_optimize)
            return f(init, *args) / ntraining

        func = partial(func_sub_theta, init_fixed)
        theta = [init]

    def step(t, opt_state, optimizer, theta):
        value, grads = value_and_grad(func)(theta, *args)
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        try:
            # norm_of_grads = jnp.sqrt(jnp.sum(jnp.square(grads)))
            norm_of_grads = jnp.linalg.norm(grads)
        except:
            norm_of_grads = sum(jnp.sum(value) for value in grads.values())
        if print_process:
            print(
                f"step{t:4} loss: {value:.4f} max_grad: {jnp.max(abs(grads)):.5f}, arg={jnp.argmax(abs(grads))}"
                # f"step{t:4} loss: {value:.4f}"
            )
            print(f"norm_of_grads: {norm_of_grads:.5f}")
            if len(theta) >= 18:
                for thet in jnp.split(theta[:18], 6):
                    print(f"{jnp.round(thet, 4)}")
            else:
                # if use_sm_kernel:
                #     print(theta)
                print(jnp.round(theta, 4))
            # print(f"theta_max: {jnp.max(theta):.5f}")
            # print(f"theta_min: {jnp.min(theta):.5f}\n")
        return value, opt_state, theta, norm_of_grads

    def doopt(init, optimizer, maxiter_GD, loss, theta, norm_of_grads_list):
        opt_state = optimizer.init(init)
        for t in range(maxiter_GD):
            value, opt_state, current_theta, norm_of_grads = step(
                t, opt_state, optimizer, theta[-1]
            )
            # keep norm of gradient, theta and loss in lists
            norm_of_grads_list.append(norm_of_grads)
            theta.append(current_theta)
            loss.append(value)
            if t < 1:
                continue
            ### TODO research stopping criterion
            # elif norm_of_grads < eps:
            #     print("converged")
            #     break
            elif jnp.abs(loss[-1] - loss[-2]) < eps:
                print("converged")
                break
            elif jnp.any(jnp.isnan(theta[-1])):
                print("diverged")
                raise Exception("発散しました、やりなおしましょう")
        return theta, loss

    res = [{"x": init}]
    # optimize by scipy method
    loss_before_optimize = func(theta[0], *args)
    print(f"loss before optimize: {loss_before_optimize}")
    loss.append(loss_before_optimize)
    print(method_scipy)
    if jnp.isnan(loss_before_optimize):
        raise Exception("初期条件でlossがnanになりました")
    # optimize by scipy
    if maxiter_scipy[0]:
        res, theta, loss = optimize_with_scipy(
            func, dfunc, hess, res, theta, loss, params_optimization, args
        )
    # optimize by sgd
    if method_GD == "adam":
        optimizer = optax.adam(lr)
    elif method_GD == "sgd":
        opt_init, opt_update, get_params = optimizers.sgd(lr)

    if maxiter_GD:
        theta, loss = doopt(
            res[-1]["x"], optimizer, maxiter_GD, loss, theta, norm_of_grads_list
        )

    return theta[-1], loss, theta, norm_of_grads_list


def optimizeParameters(f, df, hf, init, *args):
    def printer(o):
        print(
            f'\t loss = {o["fun"]:12.6e}, niter = {o["nit"]:5d}, Converged = {o["success"]:6b} : {o["message"]}'
        )
        print(f"\t\t     θ0 = {jnp.exp(init)}")
        print(f'\t\t      θ = {jnp.exp(o["x"])}')
        print(f'\t\t log(θ) = {o["x"]}')

    # maxiter:最大試行回数
    # disp:convergenceを出力するか否か
    opt = {"maxiter": 200, "disp": True}
    res = [{"x": init}]
    # _:ignore returns
    r_train, *_ = args
    ntraining = 0
    for r in r_train:
        ntraining += r.shape[0]
    # training pointsのかず
    func = lambda p, *args: f(p, *args) / ntraining
    dfunc = lambda p, *args: df(p, *args) / ntraining
    hess = lambda p, *args: hf(p, *args) / ntraining
    # 初期値を:res[-1]['x']とすることで、前回得られたhyperparameterを次回最適化の初期値として用いている
    # func(init,args)となるように定義しておく
    res.append(
        optimize.minimize(
            func, res[-1]["x"], args=args, method="Nelder-Mead", options=opt
        )
    )
    printer(res[-1])
    return res
