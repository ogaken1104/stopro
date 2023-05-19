import jax.numpy as jnp
from kernels import *


class KernelGenerator:
    def __init__(self, params_kernel):
        self.params_kernel = params_kernel
        self.dict_kernel_2d = {
            "additive": {
                "se": K_2d_SquareExp_Add,
            },
            "product": {"se": K_2d_SquareExp_Add},
        }
        self.dict_kernel_1d = {
            "se": K_1d_SquareExp,
        }
        if self.params_kernel["kernel_type"] == "rq":
            self.rq_alpha = self.params_kernel["rq_alpha"]

    def setup_kernel(
        self,
    ):
        kernel_type = self.params_kernel["kernel_type"]
        kernel_form = self.params_kernel["kernel_form"]
        input_dim = self.params_kernel["input_dim"]
        if input_dim == 1:
            kernel = self.dict_kernel_1d[kernel_type]
        elif input_dim == 2:
            kernel = self.dict_kernel_2d[kernel_form][kernel_type]
        return kernel

    def additive_kernel_2d(self, base_kernel):
        def kernel(r1, r2, theta):
            logeta, loglx, logly = theta
            return jnp.exp(logeta) * (
                base_kernel(r1[0], r2[0], loglx) + base_kernel(r1[1], r2[1], logly)
            )

        return kernel

    ### define kernels
    def K_RQ(self, x1, x2, logl):
        return jnp.power((1 + ((((x1 - x2) / logl) ** 2) / (2 * α))), -self.rq_alpha)

    def K_2d_RQ_Pro(self, r1, r2, theta):
        logeta, loglx, logly = theta
        return self.additive_kernel_2d(self.K_RQ)
