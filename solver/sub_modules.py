import jax.numpy as jnp
from jax import jacfwd, jacrev, jit


def get_init(hyperparams, kernel_type, eta=None, l=None, use_gradp_training=False, system_type='Stokes_2D'):
    eta1 = hyperparams['eta1']
    eta2 = hyperparams['eta2']
    l1 = hyperparams['l1']
    l2 = hyperparams['l2']
    alpha = hyperparams['alpha']
    if system_type == 'Stokes_2D':
        if use_gradp_training:
            θ_same = jnp.array([eta1, l1, l1])
            θ_diff = jnp.array([eta2, l2, l2])
            init = jnp.concatenate(
                [θ_same, θ_diff, θ_diff, θ_diff, θ_same, θ_diff, θ_diff, θ_same,  θ_diff, θ_same])
        else:
            if kernel_type != 'rq':
                θuxux = jnp.array([eta1, l1, l1])
                θuyuy = jnp.array([eta1, l1, l1])
                θpp = jnp.array([eta1, l1, l1])
                θuxuy = jnp.array([eta2, l2, l2])
                θuxp = jnp.array([eta2, l2, l2])
                θuyp = jnp.array([eta2, l2, l2])
            elif kernel_type == 'rq':
                θuxux = jnp.array([eta1, l1, l1, alpha])
                θuyuy = jnp.array([eta1, l1, l1, alpha])
                θpp = jnp.array([eta1, l1, l1, alpha])
                θuxuy = jnp.array([eta2, l2, l2, alpha])
                θuxp = jnp.array([eta2, l2, l2, alpha])
                θuyp = jnp.array([eta2, l2, l2, alpha])
            init = jnp.concatenate([θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp])

    elif system_type == '1D':
        init = jnp.array([eta1, l1])
    return init


def hessian(f):
    """Returns a function which computes the Hessian of a function f
           if f(x) gives the values of the function at x, and J = hessian(f)
           J(x) gives the Hessian at x"""
    return jit(jacfwd(jacrev(f)))
