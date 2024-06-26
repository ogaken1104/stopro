import jax.numpy as jnp
import numpy as np


def get_init(
    hyperparams,
    kernel_type,
    use_gradp_training=False,
    system_type="Stokes_2D",
):
    if "noise" in hyperparams:
        noise = jnp.array(hyperparams["noise"])
        del hyperparams["noise"]
    else:
        noise = None
    if kernel_type == "sm":
        for kernel_arg, hyp_each_kernel in hyperparams.items():
            for name, hyp_each in hyp_each_kernel.items():
                hyperparams[kernel_arg][name] = jnp.array(hyp_each)
        return hyperparams
    if system_type == "Stokes_3D":
        θuxux = jnp.array(hyperparams["uxux"])
        θuyuy = jnp.array(hyperparams["uyuy"])
        θuzuz = jnp.array(hyperparams["uzuz"])
        θpp = jnp.array(hyperparams["pp"])
        init = jnp.concatenate([θuxux, θuyuy, θuzuz, θpp])
    elif system_type == "Stokes_2D":
        if use_gradp_training:
            raise ValueError("Not implemented yet")
        elif len(hyperparams) == 3:
            θuxux = jnp.array(hyperparams["uxux"])
            θuyuy = jnp.array(hyperparams["uyuy"])
            θpp = jnp.array(hyperparams["pp"])
            init = jnp.concatenate([θuxux, θuyuy, θpp])
        elif len(hyperparams) == 4:
            θuxux = jnp.array(hyperparams["uxux"])
            θuyuy = jnp.array(hyperparams["uyuy"])
            θpp = jnp.array(hyperparams["pp"])
            θuxuy = jnp.array(hyperparams["uxuy"])
            init = jnp.concatenate([θuxux, θuyuy, θpp, θuxuy])
        elif len(hyperparams) == 6:
            θuxux = jnp.array(hyperparams["uxux"])
            θuyuy = jnp.array(hyperparams["uyuy"])
            θpp = jnp.array(hyperparams["pp"])
            θuxuy = jnp.array(hyperparams["uxuy"])
            θuxp = jnp.array(hyperparams["uxp"])
            θuyp = jnp.array(hyperparams["uyp"])
            init = jnp.concatenate([θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp])

    else:
        init = jnp.array(hyperparams)
    if noise:
        init = jnp.append(init, noise)
    return init


def reshape_init(init, params_model, params_kernel_arg):
    """for SM kernel, reshape initial kernel hyper-parameters"""
    num_mixture = params_model["num_mixture"]
    input_dim = params_model["input_dim"]
    init_dict = init
    init = np.zeros((6, num_mixture * (1 + 2 * input_dim)))
    for i, labl_kernel in enumerate(params_kernel_arg):
        for j, the in enumerate(init_dict[labl_kernel].values()):
            if j == 0:
                init[i, :num_mixture] = the
            elif j == 1:
                init[i, num_mixture : num_mixture * (input_dim + 1)] = the.reshape(-1)
            elif j == 2:
                init[
                    i,
                    num_mixture * (input_dim + 1) : num_mixture * (2 * input_dim + 1),
                ] = the.reshape(-1)
    init = init.reshape(-1)
    return init
