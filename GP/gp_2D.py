import jax
import jax.numpy as jnp
from jax import grad, vmap

from stopro.GP.gp import GPmodel


class GPmodel2D(GPmodel):
    """
    The base class for any 2D gaussian process model.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def outermap(f):
        return vmap(vmap(f, in_axes=(None, 0, None)), in_axes=(0, None, None))

    def setup_differential_oprators(self):
        # define operators
        def _L0(r, rp, θ):
            return jnp.sum(jnp.diag(jax.hessian(self.Kernel, 0)(r, rp, θ)))

        def _L1(r, rp, θ):
            return jnp.sum(jnp.diag(jax.hessian(self.Kernel, 1)(r, rp, θ)))

        def _d0d0(r, rp, θ):
            return jax.hessian(self.Kernel, [0, 1])(r, rp, θ)[0][1][0, 0]

        def _d0d1(r, rp, θ):
            return jax.hessian(self.Kernel, [0, 1])(r, rp, θ)[0][1][0, 1]

        def _d1d0(r, rp, θ):
            return jax.hessian(self.Kernel, [0, 1])(r, rp, θ)[0][1][1, 0]

        def _d1d1(r, rp, θ):
            return jax.hessian(self.Kernel, [0, 1])(r, rp, θ)[0][1][1, 1]

        def _d0L(r, rp, θ):
            return grad(_L1, 0)(r, rp, θ)[0]

        def _d1L(r, rp, θ):
            return grad(_L1, 0)(r, rp, θ)[1]

        def _Ld0(r, rp, θ):
            return jnp.sum(jnp.diag(jax.hessian(_d10, 0)(r, rp, θ)))

        def _Ld1(r, rp, θ):
            return jnp.sum(jnp.diag(jax.hessian(_d11, 0)(r, rp, θ)))

        def _LL(r, rp, θ):
            return jnp.sum(jnp.diag(jax.hessian(_L1, 0)(r, rp, θ)))

        def _dij(i, j, r, rp, θ):
            return grad(self.Kernel, i)(r, rp, θ)[j]

        def _d10(r, rp, θ):
            return _dij(1, 0, r, rp, θ)

        def _d11(r, rp, θ):
            return _dij(1, 1, r, rp, θ)

        def _d10(r, rp, θ):
            return grad(self.Kernel, 1)(r, rp, θ)[0]

        def _d11(r, rp, θ):
            return grad(self.Kernel, 1)(r, rp, θ)[1]

        def _d00(r, rp, θ):
            return grad(self.Kernel, 0)(r, rp, θ)[0]

        def _d01(r, rp, θ):
            return grad(self.Kernel, 0)(r, rp, θ)[1]

        self.L0 = self.outermap(_L0)
        self.L1 = self.outermap(_L1)
        self.d0d0 = self.outermap(_d0d0)
        self.d0d1 = self.outermap(_d0d1)
        self.d1d0 = self.outermap(_d1d0)
        self.d1d1 = self.outermap(_d1d1)
        self.d0L = self.outermap(_d0L)
        self.d1L = self.outermap(_d1L)
        self.Ld0 = self.outermap(_Ld0)
        self.Ld1 = self.outermap(_Ld1)
        self.LL = self.outermap(_LL)
        self.d10 = self.outermap(_d10)
        self.d11 = self.outermap(_d11)
        self.d00 = self.outermap(_d00)
        self.d01 = self.outermap(_d01)

        # define operators for rev
        def _L0_rev(r, rp, θ):
            return jnp.sum(jnp.diag(jax.hessian(self.Kernel_rev, 0)(r, rp, θ)))

        def _L1_rev(r, rp, θ):
            return jnp.sum(jnp.diag(jax.hessian(self.Kernel_rev, 1)(r, rp, θ)))

        def _d0d0_rev(r, rp, θ):
            return jax.hessian(self.Kernel_rev, [0, 1])(r, rp, θ)[0][1][0, 0]

        def _d0d1_rev(r, rp, θ):
            return jax.hessian(self.Kernel_rev, [0, 1])(r, rp, θ)[0][1][0, 1]

        def _d1d0_rev(r, rp, θ):
            return jax.hessian(self.Kernel_rev, [0, 1])(r, rp, θ)[0][1][1, 0]

        def _d1d1_rev(r, rp, θ):
            return jax.hessian(self.Kernel_rev, [0, 1])(r, rp, θ)[0][1][1, 1]

        def _d0L_rev(r, rp, θ):
            return grad(_L1_rev, 0)(r, rp, θ)[0]

        def _d1L_rev(r, rp, θ):
            return grad(_L1_rev, 0)(r, rp, θ)[1]

        def _Ld0_rev(r, rp, θ):
            return jnp.sum(jnp.diag(jax.hessian(_d10_rev, 0)(r, rp, θ)))

        def _Ld1_rev(r, rp, θ):
            return jnp.sum(jnp.diag(jax.hessian(_d11_rev, 0)(r, rp, θ)))

        def _LL_rev(r, rp, θ):
            return jnp.sum(jnp.diag(jax.hessian(_L1_rev, 0)(r, rp, θ)))

        def _dij_rev(i, j, r, rp, θ):
            return grad(self.Kernel_rev, i)(r, rp, θ)[j]

        def _d10_rev(r, rp, θ):
            return _dij_rev(1, 0, r, rp, θ)

        def _d11_rev(r, rp, θ):
            return _dij_rev(1, 1, r, rp, θ)

        def _d10_rev(r, rp, θ):
            return grad(self.Kernel_rev, 1)(r, rp, θ)[0]

        def _d11_rev(r, rp, θ):
            return grad(self.Kernel_rev, 1)(r, rp, θ)[1]

        def _d00_rev(r, rp, θ):
            return grad(self.Kernel_rev, 0)(r, rp, θ)[0]

        def _d01_rev(r, rp, θ):
            return grad(self.Kernel_rev, 0)(r, rp, θ)[1]

        self.L0_rev = self.outermap(_L0_rev)
        self.L1_rev = self.outermap(_L1_rev)
        self.d0d0_rev = self.outermap(_d0d0_rev)
        self.d0d1_rev = self.outermap(_d0d1_rev)
        self.d1d0_rev = self.outermap(_d1d0_rev)
        self.d1d1_rev = self.outermap(_d1d1_rev)
        self.d0L_rev = self.outermap(_d0L_rev)
        self.d1L_rev = self.outermap(_d1L_rev)
        self.Ld0_rev = self.outermap(_Ld0_rev)
        self.Ld1_rev = self.outermap(_Ld1_rev)
        self.LL_rev = self.outermap(_LL_rev)
        self.d10_rev = self.outermap(_d10_rev)
        self.d11_rev = self.outermap(_d11_rev)
        self.d00_rev = self.outermap(_d00_rev)
        self.d01_rev = self.outermap(_d01_rev)
