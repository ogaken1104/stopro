import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from stopro.GP.gp_2D_stokes import GPmodel2DStokes


class GPSinusoidalWithoutP(GPmodel2DStokes):
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
        kernel_type: str = None,
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
        self.kernel_type = kernel_type
        super().__init__()

    def trainingK_all(self, θ, train_pts):
        """
        Args :
            θ (jnp.array) : kernel hyperparameters
            training points (List(jnp.array)): r_ux,r_uy,r_p,r_fx,r_fy,r_div
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
        Kfxfx, Kfxfy, Kfyfy, Kfxdiv, Kfydiv, Kdivdiv = self.setup_gov_gov_kernel(θ)
        Kfxdifp, Kfydifp, Kdivdifp = self.setup_latter_difp_kernel(θ)

        Kdifuxdifux = self.setup_kernel_difdif(Kuxux)
        Kdifuxdifuy = self.setup_kernel_difdif(Kuxuy)
        Kdifuxfx = self.setup_kernel_include_difference(Kuxfx)
        Kdifuxfy = self.setup_kernel_include_difference(Kuxfy)
        Kdifuxdiv = self.setup_kernel_include_difference(Kuxdiv)
        Kdifuydifuy = self.setup_kernel_difdif(Kuyuy)
        Kdifuyfx = self.setup_kernel_include_difference(Kuyfx)
        Kdifuyfy = self.setup_kernel_include_difference(Kuyfy)
        Kdifuydiv = self.setup_kernel_include_difference(Kuydiv)

        Kdifuxp = self.setup_kernel_include_difference(Kuxp)
        Kdifuxdifp = self.setup_kernel_difdif(Kuxp)
        Kdifuyp = self.setup_kernel_include_difference(Kuyp)
        Kdifuydifp = self.setup_kernel_difdif(Kuyp)

        Kdifpdifp = self.setup_kernel_difdif(Kpp)

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

        return self.calculate_K_symmetric(train_pts, Ks)

    def mixedK_all(self, θ, test_pts, train_pts):
        """
        Args :
            θ (jnp.array) : kernel hyperparameters
            test points (List(jnp.array)): r_test
            training points (List(jnp.array)): r_ux,r_uy,r_p,r_fx,r_fy,r_div
        """
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = self.split_hyperparam(theta=θ)

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
        elif self.use_difp:
            Kuxdifp = self.setup_kernel_include_difference_prime(Kuxp)
            Kuydifp = self.setup_kernel_include_difference_prime(Kuyp)
            Kpdifp = self.setup_kernel_include_difference_prime(Kpp)
            Ks = [
                [Kuxux, Kuxuy, Kuxdifux, Kuxdifuy, Kuxfx, Kuxfy, Kuxdiv, Kuxdifp],
                [Kuxuy, Kuyuy, Kuydifux, Kuydifuy, Kuyfx, Kuyfy, Kuydiv, Kuydifp],
            ]
        else:
            Ks = [
                [Kuxux, Kuxuy, Kuxdifux, Kuxdifuy, Kuxfx, Kuxfy, Kuxdiv],
                [Kuxuy, Kuyuy, Kuydifux, Kuydifuy, Kuyfx, Kuyfy, Kuydiv],
            ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, θ, test_pts):
        """
        Args :
            θ (jnp.array) : kernel hyperparameters
            test points (List(jnp.array)): r_test
        """

        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = self.split_hyperparam(theta=θ)

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
