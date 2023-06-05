import jax
import jax.numpy as jnp
from jax import grad, vmap

from stopro.GP.gp_2D import GPmodel2D


class GPmodel2DStokes(GPmodel2D):
    def __init__(self):
        super().__init__()

    def setup_no_diffop_kernel(self, θ):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

        def Kuxux(r, rp):
            return self.K(r, rp, θuxux)

        def Kuxuy(r, rp):
            return self.K(r, rp, θuxuy)

        def Kuyuy(r, rp):
            return self.K(r, rp, θuyuy)

        def Kuxp(r, rp):
            return self.K(r, rp, θuxp)

        def Kuyp(r, rp):
            return self.K(r, rp, θuyp)

        def Kpp(r, rp):
            return self.K(r, rp, θpp)

        return Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp

    def setup_gov_gov_kernel(self, θ):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

        def Kfxfx(r, rp):
            return (
                self.d0d0(r, rp, θpp)
                - self.d0L_rev(r, rp, θuxp)
                - self.Ld0(r, rp, θuxp)
                + self.LL(r, rp, θuxux)
            )

        def Kfxfy(r, rp):
            return (
                self.d0d1(r, rp, θpp)
                - self.d0L_rev(r, rp, θuyp)
                - self.Ld1(r, rp, θuxp)
                + self.LL(r, rp, θuxuy)
            )

        def Kfyfy(r, rp):
            return (
                self.d1d1(r, rp, θpp)
                - self.d1L_rev(r, rp, θuyp)
                - self.Ld1(r, rp, θuyp)
                + self.LL(r, rp, θuyuy)
            )

        def Kfxdiv(r, rp):
            return (
                self.d0d0_rev(r, rp, θuxp)
                + self.d0d1_rev(r, rp, θuyp)
                - self.Ld0(r, rp, θuxux)
                - self.Ld1(r, rp, θuxuy)
            )

        def Kfydiv(r, rp):
            return (
                self.d1d0_rev(r, rp, θuxp)
                + self.d1d1_rev(r, rp, θuyp)
                - self.Ld0_rev(r, rp, θuxuy)
                - self.Ld1(r, rp, θuyuy)
            )

        def Kdivdiv(r, rp):
            return (
                self.d0d0(r, rp, θuxux)
                + self.d0d1(r, rp, θuxuy)
                + self.d1d0_rev(r, rp, θuxuy)
                + self.d1d1(r, rp, θuyuy)
            )

        return Kfxfx, Kfxfy, Kfyfy, Kfxdiv, Kfydiv, Kdivdiv

    def setup_nondifop_difop_kernel(self, θ):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

        def Kuxfx(r, rp):
            return self.d10(r, rp, θuxp) - self.L1(r, rp, θuxux)

        def Kuyfx(r, rp):
            return self.d10(r, rp, θuyp) - self.L1_rev(r, rp, θuxuy)

        def Kpfx(r, rp):
            return self.d10(r, rp, θpp) - self.L1_rev(r, rp, θuxp)

        def Kuxfy(r, rp):
            return self.d11(r, rp, θuxp) - self.L1(r, rp, θuxuy)

        def Kuyfy(r, rp):
            return self.d11(r, rp, θuyp) - self.L1(r, rp, θuyuy)

        def Kpfy(r, rp):
            return self.d11(r, rp, θpp) - self.L1_rev(r, rp, θuyp)

        def Kuxdiv(r, rp):
            return self.d10(r, rp, θuxux) + self.d11(r, rp, θuxuy)

        def Kuydiv(r, rp):
            return self.d10_rev(r, rp, θuxuy) + self.d11(r, rp, θuyuy)

        def Kpdiv(r, rp):
            return self.d10_rev(r, rp, θuxp) + self.d11_rev(r, rp, θuyp)

        return Kuxfx, Kuxfy, Kuxdiv, Kuyfx, Kuyfy, Kuydiv, Kpfx, Kpfy, Kpdiv

    def setup_latter_difop_kerenl(self, θ):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)
        Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp = self.setup_no_diffop_kernel(θ)

        def Kuxfx(r, rp):
            return self.d10(r, rp, θuxp) - self.L1(r, rp, θuxux)

        def Kuyfx(r, rp):
            return self.d10(r, rp, θuyp) - self.L1_rev(r, rp, θuxuy)

        # def Kpfx(r, rp):
        #     return self.d10(r, rp, θpp) - self.L1_rev(r, rp, θuxp)

        def Kuxfy(r, rp):
            return self.d11(r, rp, θuxp) - self.L1(r, rp, θuxuy)

        def Kuyfy(r, rp):
            return self.d11(r, rp, θuyp) - self.L1(r, rp, θuyuy)

        # def Kpfy(r, rp):
        #     return self.d11(r, rp, θpp) - self.L1_rev(r, rp, θuyp)

        def Kuxdiv(r, rp):
            return self.d10(r, rp, θuxux) + self.d11(r, rp, θuxuy)

        def Kuydiv(r, rp):
            return self.d10_rev(r, rp, θuxuy) + self.d11(r, rp, θuyuy)

        # def Kpdiv(r, rp):
        #     return self.d10_rev(r, rp, θuxp) + self.d11_rev(r, rp, θuyp)
        Kuxdifux = self.setup_kernel_include_difference_prime(Kuxux)
        Kuxdifuy = self.setup_kernel_include_difference_prime(Kuxuy)
        Kuydifux = self.setup_kernel_include_difference_prime(Kuxuy)
        Kuydifuy = self.setup_kernel_include_difference_prime(Kuyuy)
        Kuxdifp = self.setup_kernel_include_difference_prime(Kuxp)
        Kuydifp = self.setup_kernel_include_difference_prime(Kuyp)
        Kpdifp = self.setup_kernel_include_difference_prime(Kpp)

        return (
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
        )

    def setup_latter_difp_kernel(self, θ):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

        def Kfxp(r, rp):
            return self.d00(r, rp, θpp) - self.L0(r, rp, θuxp)

        def Kfyp(r, rp):
            return self.d01(r, rp, θpp) - self.L0(r, rp, θuyp)

        def Kdivp(r, rp):
            return self.d00(r, rp, θuxp) + self.d01(r, rp, θuyp)

        Kfxdifp = self.setup_kernel_include_difference_prime(Kfxp)
        Kfydifp = self.setup_kernel_include_difference_prime(Kfyp)
        Kdivdifp = self.setup_kernel_include_difference_prime(Kdivp)

        return Kfxdifp, Kfydifp, Kdivdifp
