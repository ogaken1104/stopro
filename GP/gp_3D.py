import jax
import jax.numpy as jnp
from jax import grad

from stopro.GP.gp_2D import GPmodel2D


class GPmodel3D(GPmodel2D):
    def __init__(self):
        super().__init__()

    def setup_differential_oprators(self):
        super().setup_differential_oprators()
        ## 3次元に必要な微分演算子を定義

        def _d0d2(r, rp, theta):
            return jax.hessian(self.Kernel, [0, 1])(r, rp, theta)[0][1][0, 2]

        def _d1d2(r, rp, theta):
            return jax.hessian(self.Kernel, [0, 1])(r, rp, theta)[0][1][1, 2]

        def _d2d2(r, rp, theta):
            return jax.hessian(self.Kernel, [0, 1])(r, rp, theta)[0][1][2, 2]

        def _d12(r, rp, theta):
            return grad(self.Kernel, 1)(r, rp, theta)[2]

        def _Ld2(r, rp, theta):
            return jnp.sum(jnp.diag(jax.hessian(_d12, 0)(r, rp, theta)))

        self.d0d2 = self.outermap(_d0d2)
        self.d1d2 = self.outermap(_d1d2)
        self.d2d2 = self.outermap(_d2d2)
        self.d12 = self.outermap(_d12)
        self.Ld2 = self.outermap(_Ld2)
