import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from stopro.GP.gp_3D_stokes_independent import GPmodel3DStokesIndependent


class GPStokes3D(GPmodel3DStokesIndependent):
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
        self.setup_all_Ks()
        # self.kernel_type = kernel_type

    def setup_trainingKs(self):
        self.trainingKs = [
            [
                self.Kuxux,
                self.Kuxuy,
                self.Kuxuz,
                self.Kuxfx,
                self.Kuxfy,
                self.Kuxfz,
                self.Kuxdiv,
            ],
            [
                self.Kuyuy,
                self.Kuyuz,
                self.Kuyfx,
                self.Kuyfy,
                self.Kuyfz,
                self.Kuydiv,
            ],
            [
                self.Kuzuz,
                self.Kuzfx,
                self.Kuzfy,
                self.Kuzfz,
                self.Kuzdiv,
            ],
            [self.Kfxfx, self.Kfxfy, self.Kfxfz, self.Kfxdiv],
            [self.Kfyfy, self.Kfyfz, self.Kfydiv],
            [self.Kfzfz, self.Kfzdiv],
            [self.Kdivdiv],
        ]

    def setup_mixedKs(self):
        if self.infer_governing_eqs:
            pass
            # self.mixedKs = [
            #     [
            #         self.Kfxux,
            #         self.Kfxuy,
            #         self.Kfxuz,
            #         self.Kfxfx,
            #         self.Kfxfy,
            #         self.Kfxdiv,
            #     ],
            #     [
            #         self.Kfyux,
            #         self.Kfyuy,
            #         self.Kfyuz,
            #         self.Kfyfx,
            #         self.Kfyfy,
            #         self.Kfydiv,
            #     ],
            #     [
            #         self.Kdivux,
            #         self.Kdivuy,
            #         self.Kdivuz,
            #         self.Kdivfx,
            #         self.Kdivfy,
            #         self.Kdivdiv,
            #     ],
            # ]
        elif self.use_difp:
            pass
        else:
            self.mixedKs = [
                [
                    self.Kuxux,
                    self.Kuxuy,
                    self.Kuxuz,
                    self.Kuxfx,
                    self.Kuxfy,
                    self.Kuxfz,
                    self.Kuxdiv,
                ],
                [
                    self.Kuxuy,
                    self.Kuyuy,
                    self.Kuyuz,
                    self.Kuyfx,
                    self.Kuyfy,
                    self.Kuyfz,
                    self.Kuydiv,
                ],
                [
                    self.Kuxuz,
                    self.Kuyuz,
                    self.Kuzuz,
                    self.Kuzfx,
                    self.Kuzfy,
                    self.Kuzfz,
                    self.Kuzdiv,
                ],
            ]

    def setup_testKs(self):
        if self.infer_governing_eqs:
            pass
            # self.testKs = [
            #     [self.Kfxfx, self.Kfxfy, self.Kfxdiv],
            #     [self.Kfyfy, self.Kfydiv],
            #     [self.Kdivdiv],
            # ]
        else:
            self.testKs = [
                [self.Kuxux, self.Kuxuy, self.Kuxuz],
                [self.Kuyuy, self.Kuyuz],
                [self.Kuzuz],
            ]
