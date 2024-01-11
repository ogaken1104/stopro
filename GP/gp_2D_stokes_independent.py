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

    ## Kernels between variables with no differential operators
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

    ## Kernels between variables with differential operators
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

    ## Kernels between variable with no differential operators and variable with
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

    ## Kernels between variable with differential operators and variable without
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

    def Kfxux(self, r, rp, theta):
        return -self.L0(r, rp, theta[self.ind_uxux])

    def Kfxuy(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kfyux(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kfyuy(self, r, rp, theta):
        return -self.L0(r, rp, theta[self.ind_uyuy])

    def Kdivfx(self, r, rp, theta):
        return -self.d0L(r, rp, theta[self.ind_uxux])

    def Kdivfy(self, r, rp, theta):
        return -self.d1L(r, rp, theta[self.ind_uyuy])

    def Kfyfx(self, r, rp, theta):
        return self.d1d0(r, rp, theta[self.ind_pp])

    ## Kernels include difference of variables
    def Kuxdifux(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kuxux)(r, rp, theta)

    def Kuxdifuy(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kuxuy)(r, rp, theta)

    def Kuydifux(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kuxuy)(r, rp, theta)

    def Kuydifuy(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kuyuy)(r, rp, theta)

    def Kuxdifp(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kuxp)(r, rp, theta)

    def Kuydifp(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kuyp)(r, rp, theta)

    def Kpdifp(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kpp)(r, rp, theta)

    def Kfxdifp(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kfxp)(r, rp, theta)

    def Kfydifp(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kfyp)(r, rp, theta)

    def Kdivdifp(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kdivp)(r, rp, theta)

    def Kdifuxdifux(self, r, rp, theta):
        return self.setup_kernel_difdif(self.Kuxux)(r, rp, theta)

    def Kdifuxdifuy(self, r, rp, theta):
        return self.setup_kernel_difdif(self.Kuxuy)(r, rp, theta)

    def Kdifuxfx(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kuxfx)(r, rp, theta)

    def Kdifuxfy(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kuxfy)(r, rp, theta)

    def Kdifuxdiv(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kuxdiv)(r, rp, theta)

    def Kdifuydifuy(self, r, rp, theta):
        return self.setup_kernel_difdif(self.Kuyuy)(r, rp, theta)

    def Kdifuyfx(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kuyfx)(r, rp, theta)

    def Kdifuyfy(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kuyfy)(r, rp, theta)

    def Kdifuydiv(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kuydiv)(r, rp, theta)

    def Kdifuxp(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kuxp)(r, rp, theta)

    def Kdifuxdifp(self, r, rp, theta):
        return self.setup_kernel_difdif(self.Kuxp)(r, rp, theta)

    def Kdifuyp(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kuyp)(r, rp, theta)

    def Kdifuydifp(self, r, rp, theta):
        return self.setup_kernel_difdif(self.Kuyp)(r, rp, theta)

    def Kdifpdifp(self, r, rp, theta):
        return self.setup_kernel_difdif(self.Kpp)(r, rp, theta)

    def Kfxdifux(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kfxux)(r, rp, theta)

    def Kfxdifuy(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kfxuy)(r, rp, theta)

    def Kfydifux(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kfyux)(r, rp, theta)

    def Kfydifuy(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kfyuy)(r, rp, theta)

    def Kdivdifux(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kdivux)(r, rp, theta)

    def Kdivdifuy(self, r, rp, theta):
        return self.setup_kernel_include_difference_prime(self.Kdivuy)(r, rp, theta)
