import jax.numpy as jnp
import optax
from jax import jit, value_and_grad
from jax.example_libraries import optimizers
from scipy import optimize

NO_GRAD_OPTIMIZE_LIST = [
    "Nelder-Mead",
    "Powell",
]

FIRST_GRAD_OPTIMIZE_LIST = ["TNC", "BFGS", "L-BFGS-B", "CG"]  # ?
SECOND_GRAD_OPTIMIZE_LIST = [
    "CG",
]


def optimize_with_scipy(func, dfunc, hess, res, theta, loss, optimize_param, args):
    def save_theta_loss(xk):
        theta.append(xk)
        loss.append(func(xk, *args))

    method_first = optimize_param["method_first"]
    maxiter_first = optimize_param["maxiter_first"]

    opt = {"maxiter": maxiter_first, "disp": 0}

    if method_first in NO_GRAD_OPTIMIZE_LIST:
        dfunc = None
        hess = None
    elif method_first in FIRST_GRAD_OPTIMIZE_LIST:
        hess = None
    elif method_first in SECOND_GRAD_OPTIMIZE_LIST:
        pass
    else:
        raise Exception("scipy optimize method is not in list")
    res.append(
        optimize.minimize(
            func,
            res[-1]["x"],
            args=args,
            method=method_first,
            jac=dfunc,
            hess=hess,
            options=opt,
            callback=save_theta_loss,
        )
    )
    print(f"loss after first optimize: {loss[-1]}, \n result of first optimize{res}")
    return res, theta, loss


def optimize_by_adam(f, df, hf, init, optimize_param, *args):
    maxiter_GD = optimize_param["maxiter_GD"]
    lr = optimize_param["lr"]
    eps = optimize_param["eps"]
    maxiter_first = optimize_param["maxiter_first"]
    method_GD = optimize_param["method_GD"]
    method_first = optimize_param["method_first"]
    print_process = optimize_param["print_process"]
    loss = []
    theta = [init]
    norm_of_grads_list = []

    def printer(loss, opt_result, init):
        print(f"\t loss = {loss[-1]:12.6e}, niter = {maxiter_GD:5d}")
        print(f"\t\t     θ0 = {jnp.exp(init)}")
        print(f"\t\t      θ = {jnp.exp(opt_result)}")
        print(f"\t\t log(θ) = {opt_result}")

    # _:ignore returns
    r_train, *_ = args
    ntraining = 0
    # opt = {"maxiter": maxiter_first, "disp": 0}
    res = [{"x": init}]
    for r in r_train:
        ntraining += r.shape[0]
    # training pointsのかず
    func = lambda p, *args: f(p, *args) / ntraining
    dfunc = lambda p, *args: df(p, *args) / ntraining
    hess = lambda p, *args: hf(p, *args) / ntraining

    def step(t, opt_state, optimizer, theta):
        value, grads = value_and_grad(func)(theta, *args)
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        norm_of_grads = jnp.sqrt(jnp.sum(jnp.square(grads)))
        if print_process:
            print(
                f"step{t:4} loss: {value:.4f} max_grad: {jnp.max(abs(grads)):.5f}, arg={jnp.argmax(abs(grads))}"
            )
            print(f"norm_of_grads: {norm_of_grads:.5f}")
            if len(theta) >= 18:
                for thet in jnp.split(theta[:18], 6):
                    print(f"{jnp.round(thet, 4)}")
            else:
                print(jnp.round(theta, 4))
            print(f"theta_max: {jnp.max(theta):.5f}")
            print(f"theta_min: {jnp.min(theta):.5f}\n")
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
            # elif jnp.abs(loss[-1]-loss[-2]) < eps:
            #     print("converged")
            #     break
            # judge convergence by norm of gradient
            elif norm_of_grads < eps:
                print("converged")
                break
            elif jnp.any(jnp.isnan(theta[-1])):
                print("diverged")
                raise Exception("発散しました、やりなおしましょう")
        return theta, loss

    # optimize by scipy method
    loss_before_optimize = func(init, *args)
    print(f"loss before optimize: {loss_before_optimize}")
    loss.append(loss_before_optimize)
    print(method_first)
    if jnp.isnan(loss_before_optimize):
        raise Exception("初期条件でlossがnanになりました")
    # optimize by scipy
    if maxiter_first:
        res, theta, loss = optimize_with_scipy(
            func, dfunc, hess, res, theta, loss, optimize_param, args
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


def optimize_by_adam_avoid_divergence(f, df, hf, init, optimize_param, *args):
    maxiter_GD = optimize_param["maxiter_GD"]
    lr = optimize_param["lr"]
    eps = optimize_param["eps"]
    maxiter_first = optimize_param["maxiter_first"]
    method_GD = optimize_param["method_GD"]
    method_first = optimize_param["method_first"]
    print(method_first)
    loss = []
    theta = [init]
    norm_of_grads_list = []

    def printer(loss, opt_result, init):
        print(f"\t loss = {loss[-1]:12.6e}, niter = {maxiter_GD:5d}")
        print(f"\t\t     θ0 = {jnp.exp(init)}")
        print(f"\t\t      θ = {jnp.exp(opt_result)}")
        print(f"\t\t log(θ) = {opt_result}")

    # _:ignore returns
    r_train, *_ = args
    ntraining = 0
    opt = {"maxiter": maxiter_first, "disp": 0}
    res = [{"x": init}]
    for r in r_train:
        ntraining += r.shape[0]
    # training pointsのかず
    func = lambda p, *args: f(p, *args) / ntraining
    dfunc = lambda p, *args: df(p, *args) / ntraining
    hess = lambda p, *args: hf(p, *args) / ntraining

    def step(t, opt_state, diverged, term_for_regularize_grads, theta_before):
        value, grads = value_and_grad(func)(get_params(opt_state), *args)
        norm_of_grads = jnp.sqrt(jnp.sum(jnp.square(grads)))
        print(
            f"step{t:4} loss: {value:.3e} max_grad: {jnp.max(abs(grads)):.3e}, arg={jnp.argmax(abs(grads))}"
        )
        # 更新したthetaの値でlossが計算不可だった場合は、gradsに倍率(term~)をかけて小さくして更新したthetaを用いる。この倍率は、場合によっては最大1まで戻され、発散する場合はどんどん小さくなる
        grads *= term_for_regularize_grads  # gradsを補正
        # 学習が遅くなりすぎないように、もし前回発散していない、かつtermが0.1以下であれば10倍する
        if not diverged and term_for_regularize_grads <= 0.1:
            term_for_regularize_grads *= 10.0
        diverged = False
        for _ in range(11):
            opt_state = opt_init(theta_before)
            opt_state = opt_update(t, grads, opt_state)
            theta = get_params(opt_state)
            # if jnp.any(jnp.isnan(theta)):
            # もし更新したthetaにおけるlossが計算できなければ、gradsを小さくしてthetaの更新をやり直す
            if jnp.isnan(func(theta, *args)):
                print("isnan")
                term_for_regularize_grads *= 0.01
                grads *= 0.01
                diverged = True
            else:
                # 次回のlossが収束していれば、ループを抜ける
                break
        print(
            f"norm_of_grads: {norm_of_grads:.3e}, term_for_regularize_grads:{term_for_regularize_grads:.2e}\n"
        )
        # for thet in jnp.split(theta, 6):
        #     print(f'{thet}')
        # print(f'theta_max: {jnp.max(theta):.5f}')
        # print(f'theta_min: {jnp.min(theta):.5f}\n')
        return value, opt_state, norm_of_grads, diverged, term_for_regularize_grads

    def doopt(init, opt_init, get_params, maxiter_GD, loss, theta, norm_of_grads_list):
        opt_state = opt_init(init)
        diverged = False
        term_for_regularize_grads = 1.0
        for t in range(maxiter_GD):
            value, opt_state, norm_of_grads, diverged, term_for_regularize_grads = step(
                t, opt_state, diverged, term_for_regularize_grads, theta[-1]
            )
            # keep norm of gradient, theta and loss in lists
            norm_of_grads_list.append(norm_of_grads)
            theta.append(get_params(opt_state))
            loss.append(value)
            if t < 1:
                continue
            # elif jnp.abs(loss[-1]-loss[-2]) < eps:
            #     print("converged")
            #     break
            # judge convergence by norm of gradient
            elif norm_of_grads < eps:
                print("converged")
                break
            elif jnp.any(jnp.isnan(theta[-1])):
                print("diverged")
                raise Exception("発散しました、やりなおしましょう")
        return theta, loss

    # optimize by scipy method
    loss_before_optimize = func(init, *args)
    print(f"loss before optimize: {loss_before_optimize}")
    loss.append(loss_before_optimize)
    if jnp.isnan(loss_before_optimize):
        raise Exception("初期条件でlossがnanになりました")
    if maxiter_first:
        if method_first == "Nelder-Mead" or method_first == "L-BFGS-B":
            res.append(
                optimize.minimize(
                    func, res[-1]["x"], args=args, method=method_first, options=opt
                )
            )
            print(
                f"loss after first optimize: {res[-1]['fun']}, \n result of first optimize{res}"
            )
        elif method_first == "TNC" or method_first == "BFGS":
            res.append(
                optimize.minimize(
                    func,
                    res[-1]["x"],
                    args=args,
                    method=method_first,
                    jac=dfunc,
                    options=opt,
                )
            )
        theta.append(res[-1]["x"])
    # optimize by sgd
    if method_GD == "adam":
        opt_init, opt_update, get_params = optimizers.adam(lr)
    elif method_GD == "sgd":
        opt_init, opt_update, get_params = optimizers.sgd(lr)

    if maxiter_GD:
        theta, loss = doopt(
            res[-1]["x"],
            opt_init,
            get_params,
            maxiter_GD,
            loss,
            theta,
            norm_of_grads_list,
        )

    return theta[-1], loss, theta, norm_of_grads_list


# fを初期条件initで各最適化方法で最適化する


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
