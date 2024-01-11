import jax.numpy as jnp
from jax import grad, vmap

from stopro.GP.gp_2D_stokes_independent import GPmodel2DStokesIndependent


class GPPoiseuilleIndependent(GPmodel2DStokesIndependent):
    def __init__(
        self,
        Kernel: callable = None,
    ):
        super().__init__()
        self.Kernel = Kernel
        self.K = self.outermap(Kernel)
        self.Kernel_rev = lambda r1, r2, θ: Kernel(r2, r1, θ)
        self.K_rev = self.outermap(self.Kernel_rev)  # really needed?
        self.setup_differential_oprators()
        self.setup_all_Ks()

    def setup_trainingKs(self):
        self.trainingKs = [
            [self.Kuxux, self.Kuxuy, self.Kuxp, self.Kuxfx, self.Kuxfy, self.Kuxdiv],
            [self.Kuyuy, self.Kuyp, self.Kuyfx, self.Kuyfy, self.Kuydiv],
            [self.Kpp, self.Kpfx, self.Kpfy, self.Kpdiv],
            [self.Kfxfx, self.Kfxfy, self.Kfxdiv],
            [self.Kfyfy, self.Kfydiv],
            [self.Kdivdiv],
        ]

    def setup_mixedKs(self):
        self.mixedKs = [
            [self.Kuxux, self.Kuxuy, self.Kuxp, self.Kuxfx, self.Kuxfy, self.Kuxdiv],
            [self.Kuyux, self.Kuyuy, self.Kuyp, self.Kuyfx, self.Kuyfy, self.Kuydiv],
            [self.Kpux, self.Kpuy, self.Kpp, self.Kpfx, self.Kpfy, self.Kpdiv],
        ]

    def setup_testKs(self):
        self.testKs = [
            [self.Kuxux, self.Kuxuy, self.Kuxp],
            [self.Kuyuy, self.Kuyp],
            [self.Kpp],
        ]
