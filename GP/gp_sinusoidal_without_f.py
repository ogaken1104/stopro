import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from stopro.GP.gp_2D import GPmodel2D


class GPSinusoidalWithoutF(GPmodel2D):
    def __init__(
        self,
        lbox: np.ndarray = None,
        use_difp: bool = False,
        use_difu: bool = False,
        infer_governing_eqs: bool = False,
        Kernel: callable = None,
        # approx_non_pd: bool = False,
    ):
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
        self.K_rev = self.Kernel_rev  # really needed?
        self.setup_differential_oprators()

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
            Kpdifp,
        ) = self.setup_latter_difop_kerenl(θ)
        _, _, _, _, _, Kdivdiv = self.setup_gov_gov_kernel(θ)
        _, _, Kdivdifp = self.setup_latter_difp_kernel(θ)

        Kdifuxdifux = self.setup_kernel_difdif(Kuxux)
        Kdifuxdifuy = self.setup_kernel_difdif(Kuxuy)
        Kdifuxdiv = self.setup_kernel_include_difference(Kuxdiv)
        Kdifuydifuy = self.setup_kernel_difdif(Kuyuy)
        Kdifuydiv = self.setup_kernel_include_difference(Kuydiv)

        Kdifuxdifp = self.setup_kernel_difdif(Kuxp)
        Kdifuydifp = self.setup_kernel_difdif(Kuyp)

        Kdifpdifp = self.setup_kernel_difdif(Kpp)

        Ks = [
            [Kuxux, Kuxuy, Kuxdifux, Kuxdifuy, Kuxdiv, Kuxdifp],
            [Kuyuy, Kuydifux, Kuydifuy, Kuydiv, Kuydifp],
            [
                Kdifuxdifux,
                Kdifuxdifuy,
                Kdifuxdiv,
                Kdifuxdifp,
            ],
            [Kdifuydifuy, Kdifuydiv, Kdifuydifp],
            [Kdivdiv, Kdivdifp],
            [Kdifpdifp],
        ]

        return self.calculate_K_symmetric(train_pts, Ks)

    def mixedK_all(self, θ, test_pts, train_pts):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

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
            Kfxdifp, Kfydifp, Kdivdifp = self.setup_latter_difp_kernel(θ)

            Kdifuxfx = self.setup_kernel_include_difference(Kuxfx)
            Kdifuxfy = self.setup_kernel_include_difference(Kuxfy)
            Kdifuxdiv = self.setup_kernel_include_difference(Kuxdiv)
            Kdifuydifuy = self.setup_kernel_difdif(Kuyuy)
            Kdifuyfx = self.setup_kernel_include_difference(Kuyfx)
            Kdifuyfy = self.setup_kernel_include_difference(Kuyfy)
            Kdifuydiv = self.setup_kernel_include_difference(Kuydiv)

            def Kfxux(r, rp):
                return self.d00_rev(r, rp, θuxp) - self.L0(r, rp, θuxux)

            def Kfxuy(r, rp):
                return self.d00_rev(r, rp, θuyp) - self.L0(r, rp, θuxuy)

            def Kfyux(r, rp):
                return self.d01_rev(r, rp, θuxp) - self.L0_rev(r, rp, θuxuy)

            def Kfyuy(r, rp):
                return self.d01_rev(r, rp, θuyp) - self.L0(r, rp, θuyuy)

            Kfxdifux = self.setup_kernel_include_difference_prime(Kfxux)
            Kfxdifuy = self.setup_kernel_include_difference_prime(Kfxuy)
            Kfydifux = self.setup_kernel_include_difference_prime(Kfyux)
            Kfydifuy = self.setup_kernel_include_difference_prime(Kfyuy)

            def Kdivux(r, rp):
                return self.d00(r, rp, θuxux) + self.d01_rev(r, rp, θuxuy)

            def Kdivuy(r, rp):
                return self.d00(r, rp, θuxuy) + self.d01(r, rp, θuyuy)

            Kdivdifux = self.setup_kernel_include_difference_prime(Kdivux)
            Kdivdifuy = self.setup_kernel_include_difference_prime(Kdivuy)

            def Kdivfx(r, rp):
                return (
                    self.d0d0(r, rp, θuxp)
                    + self.d1d0(r, rp, θuyp)
                    - self.d0L(r, rp, θuxux)
                    - self.d1L_rev(r, rp, θuxuy)
                )

            def Kdivfy(r, rp):
                return (
                    self.d0d1(r, rp, θuxp)
                    + self.d1d1(r, rp, θuyp)
                    - self.d0L(r, rp, θuxuy)
                    - self.d1L(r, rp, θuyuy)
                )

            def Kfyfx(r, rp):
                return (
                    self.d1d0(r, rp, θpp)
                    - self.d1L_rev(r, rp, θuxp)
                    - self.Ld0(r, rp, θuyp)
                    + self.LL_rev(r, rp, θuxuy)
                )

            Ks = [
                [Kfxux, Kfxuy, Kfxdifux, Kfxdifuy, Kfxfx, Kfxfy, Kfxdiv, Kfxdifp],
                [Kfyux, Kfyuy, Kfydifux, Kfydifuy, Kfyfx, Kfyfy, Kfydiv, Kfydifp],
                [
                    Kdivux,
                    Kdivuy,
                    Kdivdifux,
                    Kdivdifuy,
                    Kdivfx,
                    Kdivfy,
                    Kdivdiv,
                    Kdivdifp,
                ],
            ]
        else:
            Kuxdifp = self.setup_kernel_include_difference_prime(Kuxp)
            Kuydifp = self.setup_kernel_include_difference_prime(Kuyp)
            Kpdifp = self.setup_kernel_include_difference_prime(Kpp)
            Ks = [
                [Kuxux, Kuxuy, Kuxdifux, Kuxdifuy, Kuxdiv, Kuxdifp],
                [Kuxuy, Kuyuy, Kuydifux, Kuydifuy, Kuydiv, Kuydifp],
            ]
        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, θ, test_pts):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

        if self.infer_governing_eqs:
            Kfxfx, Kfxfy, Kfyfy, Kfxdiv, Kfydiv, Kdivdiv = self.setup_gov_gov_kernel(θ)
            Ks = [[Kfxfx, Kfxfy, Kfxdiv], [Kfyfy, Kfydiv], [Kdivdiv]]
        else:
            Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp = self.setup_no_diffop_kernel(θ)
            Ks = [
                [Kuxux, Kuxuy],
                [Kuyuy],
            ]

        return self.calculate_K_symmetric(test_pts, Ks)
