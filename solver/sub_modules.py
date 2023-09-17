import jax.numpy as jnp
from jax import jacfwd, jacrev, jit


def get_init(
    hyperparams,
    kernel_type,
    eta=None,
    l=None,
    use_gradp_training=False,
    system_type="Stokes_2D",
    params=None,
):
    if params["model"]["kernel_type"] == "sm":
        for kernel_arg, hyp_each_kernel in hyperparams.items():
            for name, hyp_each in hyp_each_kernel.items():
                hyperparams[kernel_arg][name] = jnp.array(hyp_each)
        return hyperparams
    if system_type == "Stokes_2D":
        if use_gradp_training:
            pass
            # θ_same = jnp.array([eta1, l1, l1])
            # θ_diff = jnp.array([eta2, l2, l2])
            # init = jnp.concatenate(
            #     [
            #         θ_same,
            #         θ_diff,
            #         θ_diff,
            #         θ_diff,
            #         θ_same,
            #         θ_diff,
            #         θ_diff,
            #         θ_same,
            #         θ_diff,
            #         θ_same,
            #     ]
            # )
        elif len(hyperparams) == 6:
            θuxux = jnp.array(hyperparams["uxux"])
            θuyuy = jnp.array(hyperparams["uyuy"])
            θpp = jnp.array(hyperparams["pp"])
            θuxuy = jnp.array(hyperparams["uxuy"])
            θuxp = jnp.array(hyperparams["uxp"])
            θuyp = jnp.array(hyperparams["uyp"])
            init = jnp.concatenate([θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp])
        ## when using std of noise included in training data as hyperparams
        elif len(hyperparams) == 7:
            θuxux = jnp.array(hyperparams["uxux"])
            θuyuy = jnp.array(hyperparams["uyuy"])
            θpp = jnp.array(hyperparams["pp"])
            θuxuy = jnp.array(hyperparams["uxuy"])
            θuxp = jnp.array(hyperparams["uxp"])
            θuyp = jnp.array(hyperparams["uyp"])
            θstdnoise = jnp.array(hyperparams["std_noise"])
            init = jnp.concatenate([θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp, θstdnoise])

    elif system_type == "1D":
        init = jnp.array(hyperparams)
    return init


def hessian(f):
    """Returns a function which computes the Hessian of a function f
    if f(x) gives the values of the function at x, and J = hessian(f)
    J(x) gives the Hessian at x"""
    return jit(jacfwd(jacrev(f)))
