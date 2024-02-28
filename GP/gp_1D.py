import jax
import jax.numpy as jnp
from jax import vmap

from stopro.GP.gp import GPmodel


class GPmodel1D(GPmodel):
    """
    The base class for any 1D gaussian process model.
    """

    def __init__(self):
        super().__init__()
