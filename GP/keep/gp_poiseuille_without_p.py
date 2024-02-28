import jax.numpy as jnp
from jax import grad, vmap

from stopro.GP.gp_poiseuille_independent import GPPoiseuilleIndependent


#### somehow it doesn't work, we should modify ########
class GPPoiseuilleWithoutP(GPPoiseuilleIndependent):
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
            [Kuxux, Kuxuy, Kuxfx, Kuxfy, Kuxdiv],
            [Kuyuy, Kuyfx, Kuyfy, Kuydiv],
            [Kfxfx, Kfxfy, Kfxdiv],
            [Kfyfy, Kfydiv],
            [Kdivdiv],
        ]

        return self.calculate_K_training(train_pts, Ks)

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
            [Kuxux, Kuxuy, Kuxfx, Kuxfy, Kuxdiv],
            [Kuyux, Kuyuy, Kuyfx, Kuyfy, Kuydiv],
            [Kpux, Kpuy, Kpfx, Kpfy, Kpdiv],
        ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, θ, test_pts):
        Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp = self.setup_no_diffop_kernel(θ)

        Ks = [[Kuxux, Kuxuy, Kuxp], [Kuyuy, Kuyp], [Kpp]]

        return self.calculate_K_test(test_pts, Ks)


class GPPoiseuilleInferDuxy(GPPoiseuilleWithoutP):
    def mixedK_all(self, θ, test_pts, train_pts):
        θuxux, θuyuy, θpp = self.split_hyperparam(θ)

        def Kduxyux(r, rp):
            return self.d01(r, rp, θuxux)

        def Kduxyuy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxyfx(r, rp):
            return -self.d1L(r, rp, θuxux)

        def Kduxyfy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxydiv(r, rp):
            return self.d1d0(r, rp, θuxux)

        Ks = [
            [
                Kduxyux,
                Kduxyuy,
                Kduxyfx,
                Kduxyfy,
                Kduxydiv,
            ],
        ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, θ, test_pts):
        θuxux, θuyuy, θpp = self.split_hyperparam(θ)

        def Kduxyduxy(r, rp):
            return self.d1d1(r, rp, θuxux)

        Ks = [
            [Kduxyduxy],
        ]

        return self.calculate_K_test(test_pts, Ks)
