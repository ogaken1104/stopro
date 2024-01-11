import jax.numpy as jnp
from jax import grad, vmap

from stopro.GP.gp_2D_stokes_independent import GPmodel2DStokesIndependent


#### somehow it doesn't work, we should modify ########
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

    # def trainingK_all(self, θ, train_pts):
    #     """
    #     Args :
    #     θ  : kernel hyperparameters
    #     args: training points r_ux,r_uy,r_p,r_fx,r_fy,r_div
    #     """
    #     Ks = self.trainingKs(θ)

    #     return self.calculate_K_training(train_pts, Ks)

    # def trainingKs(self, θ):
    #     Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp = self.setup_no_diffop_kernel(θ)
    #     Kfxfx, Kfxfy, Kfyfy, Kfxdiv, Kfydiv, Kdivdiv = self.setup_gov_gov_kernel(θ)
    #     (
    #         Kuxfx,
    #         Kuxfy,
    #         Kuxdiv,
    #         Kuyfx,
    #         Kuyfy,
    #         Kuydiv,
    #         Kpfx,
    #         Kpfy,
    #         Kpdiv,
    #     ) = self.setup_nondifop_difop_kernel(θ)

    #     Ks = [
    #         [Kuxux, Kuxuy, Kuxp, Kuxfx, Kuxfy, Kuxdiv],
    #         [Kuyuy, Kuyp, Kuyfx, Kuyfy, Kuydiv],
    #         [Kpp, Kpfx, Kpfy, Kpdiv],
    #         [Kfxfx, Kfxfy, Kfxdiv],
    #         [Kfyfy, Kfydiv],
    #         [Kdivdiv],
    #     ]
    #     return Ks

    def trainingK_all(self, θ, train_pts):
        """
        Args :
        θ  : kernel hyperparameters
        args: training points r_ux,r_uy,r_p,r_fx,r_fy,r_div
        """
        Ks = self.trainingKs()

        return self.calculate_K_training(train_pts, Ks, θ)

    def trainingKs(self):
        Ks = [
            [self.Kuxux, self.Kuxuy, self.Kuxp, self.Kuxfx, self.Kuxfy, self.Kuxdiv],
            [self.Kuyuy, self.Kuyp, self.Kuyfx, self.Kuyfy, self.Kuydiv],
            [self.Kpp, self.Kpfx, self.Kpfy, self.Kpdiv],
            [self.Kfxfx, self.Kfxfy, self.Kfxdiv],
            [self.Kfyfy, self.Kfydiv],
            [self.Kdivdiv],
        ]

        return Ks

    def mixedK_all(self, θ, test_pts, train_pts):
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
            return self.Kzero(r, rp, self.dummy_theta)

        def Kpux(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kpuy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        Ks = [
            [Kuxux, Kuxuy, Kuxp, Kuxfx, Kuxfy, Kuxdiv],
            [Kuyux, Kuyuy, Kuyp, Kuyfx, Kuyfy, Kuydiv],
            [Kpux, Kpuy, Kpp, Kpfx, Kpfy, Kpdiv],
        ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, θ, test_pts):
        Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp = self.setup_no_diffop_kernel(θ)

        Ks = [[Kuxux, Kuxuy, Kuxp], [Kuyuy, Kuyp], [Kpp]]

        return self.calculate_K_test(test_pts, Ks)
