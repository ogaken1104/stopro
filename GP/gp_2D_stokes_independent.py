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
        self.ind_uxux = slice(0, 3)
        self.ind_uyuy = slice(3, 6)
        self.ind_pp = slice(6, 9)

    def Kuxux(self, r, rp, theta):
        return self.K(r, rp, theta[self.ind_uxux])

    def Kuxuy(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kuyuy(self, r, rp, theta):
        return self.K(r, rp, theta[self.ind_uyuy])

    def Kuxp(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kuyp(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kpp(self, r, rp, theta):
        return self.K(r, rp, theta[self.ind_pp])

    def Kfxfx(self, r, rp, theta):
        return self.d0d0(r, rp, theta[self.ind_pp]) + self.LL(
            r, rp, theta[self.ind_uxux]
        )

    def Kfxfy(self, r, rp, theta):
        return self.d0d1(r, rp, theta[self.ind_pp])

    def Kfyfy(self, r, rp, theta):
        return self.d1d1(r, rp, theta[self.ind_pp]) + self.LL(
            r, rp, theta[self.ind_uyuy]
        )

    def Kfxdiv(self, r, rp, theta):
        return -self.Ld0(r, rp, theta[self.ind_uxux])

    def Kfydiv(self, r, rp, theta):
        return -self.Ld1(r, rp, theta[self.ind_uyuy])

    def Kdivdiv(self, r, rp, theta):
        return self.d0d0(r, rp, theta[self.ind_uxux]) + self.d1d1(
            r, rp, theta[self.ind_uyuy]
        )

    def Kuxfx(self, r, rp, theta):
        return -self.L1(r, rp, theta[self.ind_uxux])

    def Kuyfx(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kpfx(self, r, rp, theta):
        return self.d10(r, rp, theta[self.ind_pp])

    def Kuxfy(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kuyfy(self, r, rp, theta):
        return -self.L1(r, rp, theta[self.ind_uyuy])

    def Kpfy(self, r, rp, theta):
        return self.d11(r, rp, theta[self.ind_pp])

    def Kuxdiv(self, r, rp, theta):
        return self.d10(r, rp, theta[self.ind_uxux])

    def Kuydiv(self, r, rp, theta):
        return self.d11(r, rp, theta[self.ind_uyuy])

    def Kpdiv(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kuxfx(self, r, rp, theta):
        return -self.L1(r, rp, theta[self.ind_uxux])

    def Kuyfx(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kuxfy(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kuyfy(self, r, rp, theta):
        return -self.L1(r, rp, theta[self.ind_uyuy])

    def Kuxdiv(self, r, rp, theta):
        return self.d10(r, rp, theta[self.ind_uxux])

    def Kuydiv(self, r, rp, theta):
        return self.d11(r, rp, theta[self.ind_uyuy])

    def Kfxp(self, r, rp, theta):
        return self.d00(r, rp, theta[self.ind_pp])

    def Kfyp(self, r, rp, theta):
        return self.d01(r, rp, theta[self.ind_pp])

    def Kdivp(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kuyux(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kpux(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kpuy(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kuxdifux(self):
        return self.setup_kernel_include_difference_prime(self.Kuxux)

    def Kuxdifuy(self):
        return self.setup_kernel_include_difference_prime(self.Kuxuy)

    def Kuydifux(self):
        return self.setup_kernel_include_difference_prime(self.Kuxuy)

    def Kuydifuy(self):
        return self.setup_kernel_include_difference_prime(self.Kuyuy)

    def Kuxdifp(self):
        return self.setup_kernel_include_difference_prime(self.Kuxp)

    def Kuydifp(self):
        return self.setup_kernel_include_difference_prime(self.Kuyp)

    def Kpdifp(self):
        return self.setup_kernel_include_difference_prime(self.Kpp)

    def Kfxdifp(self):
        return self.setup_kernel_include_difference_prime(self.Kfxp)

    def Kfydifp(self):
        return self.setup_kernel_include_difference_prime(self.Kfyp)

    def Kdivdifp(self):
        return self.setup_kernel_include_difference_prime(self.Kdivp)
