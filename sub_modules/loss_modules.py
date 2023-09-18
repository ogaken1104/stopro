import jax.numpy as jnp
from jax import jacfwd, jacrev, jit


def logposterior(loglikelihood, params_optimization):
    if params_optimization["loss_ridge_regression"]:
        return (
            lambda θ, *args: loglikelihood(θ, *args)
            + jnp.sum(θ)
            + params_optimization["ridge_alpha"] * jnp.sum(jnp.square(jnp.exp(θ)))
        )
    else:
        return lambda θ, *args: loglikelihood(θ, *args) + jnp.sum(θ)


def hessian(f):
    """Returns a function which computes the Hessian of a function f
    if f(x) gives the values of the function at x, and J = hessian(f)
    J(x) gives the Hessian at x"""
    return jit(jacfwd(jacrev(f)))
