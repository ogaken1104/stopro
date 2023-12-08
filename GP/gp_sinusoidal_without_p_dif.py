import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from stopro.GP.gp_sinusoidal_independent import GPSinusoidalWithoutPIndependent


class GPSinusoidalWithoutPDif(GPSinusoidalWithoutPIndependent):
    def trainingKs(self, θ):
        Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp = self.setup_no_diffop_kernel(θ)
        (
            Kuxfx,
            Kuyfx,
            Kuxfy,
            Kuyfy,
            Kuxdiv,
            Kuydiv,
            Kuxdifux,
            Kuxdifuy,
            Kuydifux,
            Kuydifuy,
            Kuxdifp,
            Kuydifp,
            Kpdifp,
        ) = self.setup_latter_difop_kerenl(θ)
        Kfxfx, Kfxfy, Kfyfy, Kfxdiv, Kfydiv, Kdivdiv = self.setup_gov_gov_kernel(θ)

        Ks = [
            [Kuxux, Kuxuy, Kuxfx, Kuxfy, Kuxdiv],
            [Kuyuy, Kuyfx, Kuyfy, Kuydiv],
            [Kfxfx, Kfxfy, Kfxdiv],
            [Kfyfy, Kfydiv],
            [Kdivdiv],
        ]
        return Ks

    def mixedK_all(self, θ, test_pts, train_pts):
        """
        Args :
            θ (jnp.array) : kernel hyperparameters
            test points (List(jnp.array)): r_test
            training points (List(jnp.array)): r_ux,r_uy,r_p,r_fx,r_fy,r_div
        """
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)

        Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp = self.setup_no_diffop_kernel(θ)
        (
            Kuxfx,
            Kuyfx,
            Kuxfy,
            Kuyfy,
            Kuxdiv,
            Kuydiv,
            Kuxdifux,
            Kuxdifuy,
            Kuydifux,
            Kuydifuy,
            Kuxdifp,
            Kuydifp,
            Kpdifp,
        ) = self.setup_latter_difop_kerenl(θ)

        if self.infer_governing_eqs:
            Kfxfx, Kfxfy, Kfyfy, Kfxdiv, Kfydiv, Kdivdiv = self.setup_gov_gov_kernel(θ)

            def Kfxux(r, rp):
                return -self.L0(r, rp, θuxux)

            def Kfxuy(r, rp):
                return self.Kzero(r, rp, self.dummy_theta)

            def Kfyux(r, rp):
                return self.Kzero(r, rp, self.dummy_theta)

            def Kfyuy(r, rp):
                return -self.L0(r, rp, θuyuy)

            def Kdivux(r, rp):
                return self.d00(r, rp, θuxux)

            def Kdivuy(r, rp):
                return self.d01(r, rp, θuyuy)

            def Kdivfx(r, rp):
                return -self.d0L(r, rp, θuxux)

            def Kdivfy(r, rp):
                return -self.d1L(r, rp, θuyuy)

            def Kfyfx(r, rp):
                return self.d1d0(r, rp, θpp)

            Ks = [
                [Kfxux, Kfxuy, Kfxfx, Kfxfy, Kfxdiv],
                [Kfyux, Kfyuy, Kfyfx, Kfyfy, Kfydiv],
                [
                    Kdivux,
                    Kdivuy,
                    Kdivfx,
                    Kdivfy,
                    Kdivdiv,
                ],
            ]
        elif self.use_difp:
            Kuxdifp = self.setup_kernel_include_difference_prime(Kuxp)
            Kuydifp = self.setup_kernel_include_difference_prime(Kuyp)
            Kpdifp = self.setup_kernel_include_difference_prime(Kpp)
            Ks = [
                [Kuxux, Kuxuy, Kuxfx, Kuxfy, Kuxdiv],
                [Kuxuy, Kuyuy, Kuyfx, Kuyfy, Kuydiv],
            ]
        else:
            Ks = [
                [Kuxux, Kuxuy, Kuxfx, Kuxfy, Kuxdiv],
                [Kuxuy, Kuyuy, Kuyfx, Kuyfy, Kuydiv],
            ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)
