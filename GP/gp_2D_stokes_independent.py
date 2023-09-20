import jax.numpy as jnp

from stopro.GP.gp_2D import GPmodel2D


class GPmodel2DStokesIndependent(GPmodel2D):
    """
    The base class for 2D stokes gaussian process
    """

    def __init__(self):
        super().__init__()
        self.split_hyperparam = lambda theta: jnp.split(theta, 3)
        self.Kernel_zero = lambda r, rp, theta: 0.0
        self.Kzero = self.outermap(self.Kernel_zero)
        self.dummy_theta = jnp.zeros(3)

    def setup_no_diffop_kernel(self, θ):
        """
        Construct kernel of (uxux, uxuy, uyuy, uxp, uyp, pp)

        Args:
            \theta (jnp.array): array of hyper-parameter

        Returns:
            Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp
        """
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)

        def Kuxux(r, rp):
            return self.K(r, rp, θuxux)

        def Kuxuy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kuyuy(r, rp):
            return self.K(r, rp, θuyuy)

        def Kuxp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kuyp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kpp(r, rp):
            return self.K(r, rp, θpp)

        return Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp

    def setup_gov_gov_kernel(self, θ):
        """
        Construct kernels of (Kfxfx, Kfxfy, Kfyfy, Kfxdiv, Kfydiv, Kdivdiv)

        Args:
            \theta (jnp.array): array of hyper-parameter

        Returns:
            Kfxfx, Kfxfy, Kfyfy, Kfxdiv, Kfydiv, Kdivdiv
        """
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)

        def Kfxfx(r, rp):
            return self.d0d0(r, rp, θpp) + self.LL(r, rp, θuxux)

        def Kfxfy(r, rp):
            return self.d0d1(r, rp, θpp)

        def Kfyfy(r, rp):
            return self.d1d1(r, rp, θpp) + self.LL(r, rp, θuyuy)

        def Kfxdiv(r, rp):
            return -self.Ld0(r, rp, θuxux)

        def Kfydiv(r, rp):
            return -self.Ld1(r, rp, θuyuy)

        def Kdivdiv(r, rp):
            return self.d0d0(r, rp, θuxux) + self.d1d1(r, rp, θuyuy)

        return Kfxfx, Kfxfy, Kfyfy, Kfxdiv, Kfydiv, Kdivdiv

    def setup_nondifop_difop_kernel(self, θ):
        """
        Construct kernels of (Kuxfx, Kuxfy, Kuxdiv, Kuyfx, Kuyfy, Kuydiv, Kpfx, Kpfy, Kpdiv)

        Args:
            \theta (jnp.array): array of hyper-parameter

        Returns:
            Kuxfx, Kuxfy, Kuxdiv, Kuyfx, Kuyfy, Kuydiv, Kpfx, Kpfy, Kpdiv
        """
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)

        def Kuxfx(r, rp):
            return -self.L1(r, rp, θuxux)

        def Kuyfx(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kpfx(r, rp):
            return self.d10(r, rp, θpp)

        def Kuxfy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kuyfy(r, rp):
            return -self.L1(r, rp, θuyuy)

        def Kpfy(r, rp):
            return self.d11(r, rp, θpp)

        def Kuxdiv(r, rp):
            return self.d10(r, rp, θuxux)

        def Kuydiv(r, rp):
            return self.d11(r, rp, θuyuy)

        def Kpdiv(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        return Kuxfx, Kuxfy, Kuxdiv, Kuyfx, Kuyfy, Kuydiv, Kpfx, Kpfy, Kpdiv

    def setup_latter_difop_kerenl(self, θ):
        """
        Construct kernels only the latter of arguments is shifted difference

        Args:
            \theta (jnp.array): array of hyper-parameter

        Returns:
            Kernels,
        """
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)
        Kuxux, Kuxuy, Kuyuy, Kuxp, Kuyp, Kpp = self.setup_no_diffop_kernel(θ)

        def Kuxfx(r, rp):
            return -self.L1(r, rp, θuxux)

        def Kuyfx(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        # def Kpfx(r, rp):
        #     return self.d10(r, rp, θpp) - self.L1_rev(r, rp, θuxp)

        def Kuxfy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kuyfy(r, rp):
            return -self.L1(r, rp, θuyuy)

        # def Kpfy(r, rp):
        #     return self.d11(r, rp, θpp) - self.L1_rev(r, rp, θuyp)

        def Kuxdiv(r, rp):
            return self.d10(r, rp, θuxux)

        def Kuydiv(r, rp):
            return self.d11(r, rp, θuyuy)

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
        """
        Construct kernels of (Kfxdifp, Kfydifp, Kdivdifp)

        Args:
            \theta (jnp.array): array of hyper-parameter

        Returns:
            Kfxdifp, Kfydifp, Kdivdifp
        """
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)

        def Kfxp(r, rp):
            return self.d00(r, rp, θpp)

        def Kfyp(r, rp):
            return self.d01(r, rp, θpp)

        def Kdivp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        Kfxdifp = self.setup_kernel_include_difference_prime(Kfxp)
        Kfydifp = self.setup_kernel_include_difference_prime(Kfyp)
        Kdivdifp = self.setup_kernel_include_difference_prime(Kdivp)

        return Kfxdifp, Kfydifp, Kdivdifp
