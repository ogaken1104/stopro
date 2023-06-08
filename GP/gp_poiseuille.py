import jax.numpy as jnp
from jax import grad, vmap

from stopro.GP.gp_2D_stokes import GPmodel2DStokes


#### somehow it doesn't work, we should modify ########
class GPPoiseuille(GPmodel2DStokes):
    def __init__(
        self,
        Kernel: callable = None,
        normalize_covariance_matrix: bool = False,
    ):
        super().__init__()
        self.Kernel = Kernel
        self.K = self.outermap(Kernel)
        self.Kernel_rev = lambda r1, r2, θ: Kernel(r2, r1, θ)
        self.K_rev = self.outermap(self.Kernel_rev)  # really needed?
        self.setup_differential_oprators()
        self.normalize_covariance_matrix = normalize_covariance_matrix

    def trainingK_all(self, θ, train_pts):
        """
        Args :
        θ  : kernel hyperparameters
        args: training points r_ux,r_uy,r_p,r_fx,r_fy,r_div
        """

        Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp = self.setup_no_diffop_kernel(θ)
        Kfxfx, Kfxfy, Kfyfy, Kfxdiv, Kfydiv, Kdivdiv = self.setup_gov_gov_kernel(θ)
        (
            Kuxfx,
            Kuxfy,
            Kuxdiv,
            Kuyfx,
            Kuyfy,
            Kuydiv,
            Kpfx,
            Kpfy,
            Kpdiv,
        ) = self.setup_nondifop_difop_kernel(θ)

        Ks = [
            [Kuxux, Kuxuy, Kuxp, Kuxfx, Kuxfy, Kuxdiv],
            [Kuyuy, Kuyp, Kuyfx, Kuyfy, Kuydiv],
            [Kpp, Kpfx, Kpfy, Kpdiv],
            [Kfxfx, Kfxfy, Kfxdiv],
            [Kfyfy, Kfydiv],
            [Kdivdiv],
        ]

        return self.calculate_K_symmetric(train_pts, Ks)

    def mixedK_all(self, θ, test_pts, train_pts):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)
        Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp = self.setup_no_diffop_kernel(θ)
        (
            Kuxfx,
            Kuxfy,
            Kuxdiv,
            Kuyfx,
            Kuyfy,
            Kuydiv,
            Kpfx,
            Kpfy,
            Kpdiv,
        ) = self.setup_nondifop_difop_kernel(θ)

        def Kuyux(r, rp):
            return self.K_rev(r, rp, θuxuy)

        def Kpux(r, rp):
            return self.K_rev(r, rp, θuxp)

        def Kpuy(r, rp):
            return self.K_rev(r, rp, θuyp)

        Ks = [
            [Kuxux, Kuxuy, Kuxp, Kuxfx, Kuxfy, Kuxdiv],
            [Kuyux, Kuyuy, Kuyp, Kuyfx, Kuyfy, Kuydiv],
            [Kpux, Kpuy, Kpp, Kpfx, Kpfy, Kpdiv],
        ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, θ, test_pts):
        Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp = self.setup_no_diffop_kernel(θ)

        Ks = [[Kuxux, Kuxuy, Kuxp], [Kuyuy, Kuyp], [Kpp]]

        return self.calculate_K_symmetric(test_pts, Ks)
