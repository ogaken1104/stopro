import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from stopro.GP.gp_sinusoidal_without_p import GPSinusoidalWithoutP


class GPSinusoidalInferDifP(GPSinusoidalWithoutP):
    def trainingK_all(self, θ, train_pts):
        """
        Args :
        θ  : kernel hyperparameters
        args: training points r_ux,r_uy,r_p,r_fx,r_fy,r_div
        """

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
            _,
        ) = self.setup_latter_difop_kerenl(θ)
        Kfxfx, Kfxfy, Kfyfy, Kfxdiv, Kfydiv, Kdivdiv = self.setup_gov_gov_kernel(θ)

        Kdifuxdifux = self.setup_kernel_difdif(Kuxux)
        Kdifuxdifuy = self.setup_kernel_difdif(Kuxuy)
        Kdifuxfx = self.setup_kernel_include_difference(Kuxfx)
        Kdifuxfy = self.setup_kernel_include_difference(Kuxfy)
        Kdifuxdiv = self.setup_kernel_include_difference(Kuxdiv)
        Kdifuydifuy = self.setup_kernel_difdif(Kuyuy)
        Kdifuyfx = self.setup_kernel_include_difference(Kuyfx)
        Kdifuyfy = self.setup_kernel_include_difference(Kuyfy)
        Kdifuydiv = self.setup_kernel_include_difference(Kuydiv)

        if self.use_difp:
            Kfxdifp, Kfydifp, Kdivdifp = self.setup_latter_difp_kernel(θ)
            Kdifpdifp = self.setup_kernel_difdif(Kpp)
            Kdifuydifp = self.setup_kernel_difdif(Kuyp)
            Kdifuxdifp = self.setup_kernel_difdif(Kuxp)
            Ks = [
                [Kuxux, Kuxuy, Kuxdifux, Kuxdifuy, Kuxfx, Kuxfy, Kuxdiv, Kuxdifp],
                [Kuyuy, Kuydifux, Kuydifuy, Kuyfx, Kuyfy, Kuydiv, Kuydifp],
                [
                    Kdifuxdifux,
                    Kdifuxdifuy,
                    Kdifuxfx,
                    Kdifuxfy,
                    Kdifuxdiv,
                    Kdifuxdifp,
                ],
                [Kdifuydifuy, Kdifuyfx, Kdifuyfy, Kdifuydiv, Kdifuydifp],
                [Kfxfx, Kfxfy, Kfxdiv, Kfxdifp],
                [Kfyfy, Kfydiv, Kfydifp],
                [Kdivdiv, Kdivdifp],
                [Kdifpdifp],
            ]
        else:
            Ks = [
                [Kuxux, Kuxuy, Kuxdifux, Kuxdifuy, Kuxfx, Kuxfy, Kuxdiv],
                [Kuyuy, Kuydifux, Kuydifuy, Kuyfx, Kuyfy, Kuydiv],
                [
                    Kdifuxdifux,
                    Kdifuxdifuy,
                    Kdifuxfx,
                    Kdifuxfy,
                    Kdifuxdiv,
                ],
                [Kdifuydifuy, Kdifuyfx, Kdifuyfy, Kdifuydiv],
                [Kfxfx, Kfxfy, Kfxdiv],
                [Kfyfy, Kfydiv],
                [Kdivdiv],
            ]

        return self.calculate_K_symmetric(train_pts, Ks)

    def mixedK_all(self, θ, test_pts, train_pts):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

        def Kpux(r, rp):
            return self.K_rev(r, rp, θuxp)

        def Kpuy(r, rp):
            return self.K_rev(r, rp, θuyp)

        def Kpfx(r, rp):
            return self.d10(r, rp, θpp) - self.L1_rev(r, rp, θuxp)

        def Kpfy(r, rp):
            return self.d11(r, rp, θpp) - self.L1_rev(r, rp, θuyp)

        def Kpdiv(r, rp):
            return self.d10_rev(r, rp, θuxp) + self.d11_rev(r, rp, θuyp)

        Kdifpux = self.setup_kernel_include_difference(Kpux)
        Kdifpuy = self.setup_kernel_include_difference(Kpuy)
        Kdifpfx = self.setup_kernel_include_difference(Kpfx)
        Kdifpfy = self.setup_kernel_include_difference(Kpfy)
        Kdifpdiv = self.setup_kernel_include_difference(Kpdiv)

        Kdifpdifux = self.setup_kernel_difdif(Kpux)
        Kdifpdifuy = self.setup_kernel_difdif(Kpuy)

        if self.use_difp:
            _, _, _, _, _, Kpp = self.setup_no_diffop_kernel(θ)
            Kdifpdifp = self.setup_kernel_difdif(Kpp)
            Ks = [
                [
                    Kdifpux,
                    Kdifpuy,
                    Kdifpdifux,
                    Kdifpdifuy,
                    Kdifpfx,
                    Kdifpfy,
                    Kdifpdiv,
                    Kdifpdifp,
                ],
            ]
        else:
            Ks = [
                [Kdifpux, Kdifpuy, Kdifpdifux, Kdifpdifuy, Kdifpfx, Kdifpfy, Kdifpdiv],
            ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, θ, test_pts):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

        Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp = self.setup_no_diffop_kernel(θ)
        Kdifpdifp = self.setup_kernel_difdif(Kpp)

        Ks = [
            [Kdifpdifp],
        ]

        return self.calculate_K_symmetric(test_pts, Ks)
