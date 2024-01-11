import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from stopro.GP.gp_2D_stokes_independent import GPmodel2DStokesIndependent


class GPSinusoidalWithoutPIndependent(GPmodel2DStokesIndependent):
    """
    Class for the inference using difference of pressure in 2D Stokes system.
    """

    def __init__(
        self,
        lbox: np.ndarray = None,
        use_difp: bool = False,
        use_difu: bool = False,
        infer_governing_eqs: bool = False,
        Kernel: callable = None,
        index_optimize_noise: list = None,
        # kernel_type: str = None,
        # approx_non_pd: bool = False,
    ):
        """
        Args:
            lbox (np.ndarray): periodic length
            use_difp (bool): if using difference of pressure at inlet and outlet as training points
            use_difu (bool): if using difference of velocity at inlet and outlet as training points
            infer_governing_eqs (bool): if infer governing equation instead of velocity (and pressure)
            Kernel (callable): kernel function to use
        """
        super().__init__()
        self.lbox = lbox
        self.use_difp = use_difp
        self.use_difu = use_difu
        self.infer_governing_eqs = infer_governing_eqs
        self.Kernel = Kernel
        self.K = self.outermap(Kernel)
        # def Kernel_rev(r1, r2, θ):
        #     return Kernel(r2, r1, θ)
        self.Kernel_rev = lambda r1, r2, θ: Kernel(r2, r1, θ)
        self.K_rev = self.outermap(self.Kernel_rev)  # really needed?
        self.setup_differential_oprators()
        self.index_optimize_noise = index_optimize_noise
        # self.kernel_type = kernel_type

    def trainingKs(self):
        Ks = [
            [
                self.Kuxux,
                self.Kuxuy,
                self.Kuxdifux,
                self.Kuxdifuy,
                self.Kuxfx,
                self.Kuxfy,
                self.Kuxdiv,
                self.Kuxdifp,
            ],
            [
                self.Kuyuy,
                self.Kuydifux,
                self.Kuydifuy,
                self.Kuyfx,
                self.Kuyfy,
                self.Kuydiv,
                self.Kuydifp,
            ],
            [
                self.Kdifuxdifux,
                self.Kdifuxdifuy,
                self.Kdifuxfx,
                self.Kdifuxfy,
                self.Kdifuxdiv,
                self.Kdifuxdifp,
            ],
            [
                self.Kdifuydifuy,
                self.Kdifuyfx,
                self.Kdifuyfy,
                self.Kdifuydiv,
                self.Kdifuydifp,
            ],
            [self.Kfxfx, self.Kfxfy, self.Kfxdiv, self.Kfxdifp],
            [self.Kfyfy, self.Kfydiv, self.Kfydifp],
            [self.Kdivdiv, self.Kdivdifp],
            [self.Kdifpdifp],
        ]
        return Ks

    def mixedKs(self):
        if self.infer_governing_eqs:
            Ks = [
                [
                    self.Kfxux,
                    self.Kfxuy,
                    self.Kfxdifux,
                    self.Kfxdifuy,
                    self.Kfxfx,
                    self.Kfxfy,
                    self.Kfxdiv,
                    self.Kfxdifp,
                ],
                [
                    self.Kfyux,
                    self.Kfyuy,
                    self.Kfydifux,
                    self.Kfydifuy,
                    self.Kfyfx,
                    self.Kfyfy,
                    self.Kfydiv,
                    self.Kfydifp,
                ],
                [
                    self.Kdivux,
                    self.Kdivuy,
                    self.Kdivdifux,
                    self.Kdivdifuy,
                    self.Kdivfx,
                    self.Kdivfy,
                    self.Kdivdiv,
                    self.Kdivdifp,
                ],
            ]
        elif self.use_difp:
            Ks = [
                [
                    self.Kuxux,
                    self.Kuxuy,
                    self.Kuxdifux,
                    self.Kuxdifuy,
                    self.Kuxfx,
                    self.Kuxfy,
                    self.Kuxdiv,
                    self.Kuxdifp,
                ],
                [
                    self.Kuxuy,
                    self.Kuyuy,
                    self.Kuydifux,
                    self.Kuydifuy,
                    self.Kuyfx,
                    self.Kuyfy,
                    self.Kuydiv,
                    self.Kuydifp,
                ],
            ]
        else:
            Ks = [
                [
                    self.Kuxux,
                    self.Kuxuy,
                    self.Kuxdifux,
                    self.Kuxdifuy,
                    self.Kuxfx,
                    self.Kuxfy,
                    self.Kuxdiv,
                ],
                [
                    self.Kuxuy,
                    self.Kuyuy,
                    self.Kuydifux,
                    self.Kuydifuy,
                    self.Kuyfx,
                    self.Kuyfy,
                    self.Kuydiv,
                ],
            ]

        return Ks

    def testKs(self):
        if self.infer_governing_eqs:
            Ks = [
                [self.Kfxfx, self.Kfxfy, self.Kfxdiv],
                [self.Kfyfy, self.Kfydiv],
                [self.Kdivdiv],
            ]
        else:
            Ks = [
                [self.Kuxux, self.Kuxuy],
                [self.Kuyuy],
            ]

        return Ks
