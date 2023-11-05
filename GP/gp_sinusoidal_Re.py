import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from stopro.GP.gp_2D_Re import GPmodel2DStokesRe


class GPSinusoidalRe(GPmodel2DStokesRe):
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
        Re: float = 1.0,
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
        super().__init__(Re=Re)
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

    # def setup_kernel_difdif(self, K_func):
    #     """
    #     Function that construct a kernel that calculates shifted difference of variable both at first and second argument.
    #     S_{L\boldsymbol{e}^{alpha}}S^{prime}_{L\boldsymbol{e}^{alpha}}(kernel)
    #     """

    #     def K_difdif(r, rp):
    #         return (
    #             K_func(r + self.lbox, rp + self.lbox)
    #             - K_func(r + self.lbox, rp)
    #             - K_func(r, rp + self.lbox)
    #             + K_func(r, rp)
    #         ) / 2.0

    #     return K_difdif

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

        return self.calculate_K_training(train_pts, Ks)

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
            Kfxdifp, Kfydifp, Kdivdifp = self.setup_latter_difp_kernel(θ)

            Kdifuxfx = self.setup_kernel_include_difference(Kuxfx)
            Kdifuxfy = self.setup_kernel_include_difference(Kuxfy)
            Kdifuxdiv = self.setup_kernel_include_difference(Kuxdiv)
            Kdifuydifuy = self.setup_kernel_difdif(Kuyuy)
            Kdifuyfx = self.setup_kernel_include_difference(Kuyfx)
            Kdifuyfy = self.setup_kernel_include_difference(Kuyfy)
            Kdifuydiv = self.setup_kernel_include_difference(Kuydiv)

            def Kfxux(r, rp):
                return -self.L0(r, rp, θuxux) / self.Re

            def Kfxuy(r, rp):
                return self.Kzero(r, rp, self.dummy_theta)

            def Kfyux(r, rp):
                return self.Kzero(r, rp, self.dummy_theta)

            def Kfyuy(r, rp):
                return -self.L0(r, rp, θuyuy) / self.Re

            Kfxdifux = self.setup_kernel_include_difference_prime(Kfxux)
            Kfxdifuy = self.setup_kernel_include_difference_prime(Kfxuy)
            Kfydifux = self.setup_kernel_include_difference_prime(Kfyux)
            Kfydifuy = self.setup_kernel_include_difference_prime(Kfyuy)

            def Kdivux(r, rp):
                return self.d00(r, rp, θuxux)

            def Kdivuy(r, rp):
                return self.d01(r, rp, θuyuy)

            Kdivdifux = self.setup_kernel_include_difference_prime(Kdivux)
            Kdivdifuy = self.setup_kernel_include_difference_prime(Kdivuy)

            def Kdivfx(r, rp):
                return -self.d0L(r, rp, θuxux) / self.Re

            def Kdivfy(r, rp):
                return -self.d1L(r, rp, θuyuy) / self.Re

            def Kfyfx(r, rp):
                return self.d1d0(r, rp, θpp)

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

        if self.infer_governing_eqs:
            Kfxfx, Kfxfy, Kfyfy, Kfxdiv, Kfydiv, Kdivdiv = self.setup_gov_gov_kernel(θ)
            Ks = [[Kfxfx, Kfxfy, Kfxdiv], [Kfyfy, Kfydiv], [Kdivdiv]]
        else:
            Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp = self.setup_no_diffop_kernel(θ)
            Ks = [
                [Kuxux, Kuxuy],
                [Kuyuy],
            ]

        return self.calculate_K_test(test_pts, Ks)
