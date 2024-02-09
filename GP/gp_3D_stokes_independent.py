import jax.numpy as jnp

from stopro.GP.gp_3D import GPmodel3D
from stopro.GP.gp_2D_stokes_independent import GPmodel2DStokesIndependent


class GPmodel3DStokesIndependent(GPmodel3D, GPmodel2DStokesIndependent):
    """
    The base class for 2D stokes gaussian process
    """

    def __init__(self):
        GPmodel3D.__init__(self)
        self.split_hyperparam = lambda theta: jnp.split(theta, 4)
        self.Kernel_zero = lambda r, rp, theta: 0.0
        self.Kzero = self.outermap(self.Kernel_zero)
        self.dummy_theta = jnp.zeros(3)
        self.ind_uxux = slice(0, 4)
        self.ind_uyuy = slice(4, 8)
        self.ind_uzuz = slice(8, 12)
        self.ind_pp = slice(12, 16)

    ## Kernels between variables with no differential operators

    def Kuxuz(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kuyuz(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kuzuz(self, r, rp, theta):
        return self.K(r, rp, theta[self.ind_uzuz])

    def Kuzp(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    ## Kernels between variables with differential operators

    def Kfxfz(self, r, rp, theta):
        return self.d0d2(r, rp, theta[self.ind_pp])

    def Kfyfz(self, r, rp, theta):
        return self.d1d2(r, rp, theta[self.ind_pp])

    def Kfzfz(self, r, rp, theta):
        return self.d2d2(r, rp, theta[self.ind_pp]) + self.LL(
            r, rp, theta[self.ind_uzuz]
        )

    def Kfzdiv(self, r, rp, theta):
        return -self.Ld2(r, rp, theta[self.ind_uzuz])

    def Kdivdiv(self, r, rp, theta):
        return (
            self.d0d0(r, rp, theta[self.ind_uxux])
            + self.d1d1(r, rp, theta[self.ind_uyuy])
            + self.d2d2(r, rp, theta[self.ind_uzuz])
        )

    ## Kernels between variable with no differential operators and variable with
    def Kuxfz(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kuyfz(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kuzfx(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kuzfy(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kuzfz(self, r, rp, theta):
        return -self.L1(r, rp, theta[self.ind_uzuz])

    def Kuzdiv(self, r, rp, theta):
        return self.d12(r, rp, theta[self.ind_uzuz])
    
    ## Kernels for infer difp
    def Kpux(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kpuy(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kpuz(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)
    
    def Kpfx(self, r, rp, theta):
        return self.d10(r, rp, theta[self.ind_pp])

    def Kpfy(self, r, rp, theta):
        return self.d11(r, rp, theta[self.ind_pp])

    def Kpfz(self, r, rp, theta):
        return self.d12(r, rp, theta[self.ind_pp])
    
    def Kpdiv(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)
    
    def Kdifpux(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kpux)(r, rp, theta)

    def Kdifpuy(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kpuy)(r, rp, theta)

    def Kdifpuz(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kpuz)(r, rp, theta)
    
    def Kdifpfx(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kpfx)(r, rp, theta)

    def Kdifpfy(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kpfy)(r, rp, theta)

    def Kdifpfz(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kpfz)(r, rp, theta)
    
    def Kdifpdiv(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kpdiv)(r, rp, theta)

    # ## Kernels between variable with differential operators and variable without (used for inference of governing eqs.)
    # def Kfxp(self, r, rp, theta):
    #     return self.d00(r, rp, theta[self.ind_pp])

    # def Kfyp(self, r, rp, theta):
    #     return self.d01(r, rp, theta[self.ind_pp])

    # def Kdivp(self, r, rp, theta):
    #     return self.Kzero(r, rp, self.dummy_theta)

    # def Kuyux(self, r, rp, theta):
    #     return self.Kzero(r, rp, self.dummy_theta)

    # def Kpux(self, r, rp, theta):
    #     return self.Kzero(r, rp, self.dummy_theta)

    # def Kpuy(self, r, rp, theta):
    #     return self.Kzero(r, rp, self.dummy_theta)

    # def Kfxux(self, r, rp, theta):
    #     return -self.L0(r, rp, theta[self.ind_uxux])

    # def Kfxuy(self, r, rp, theta):
    #     return self.Kzero(r, rp, self.dummy_theta)

    # def Kfyux(self, r, rp, theta):
    #     return self.Kzero(r, rp, self.dummy_theta)

    # def Kfyuy(self, r, rp, theta):
    #     return -self.L0(r, rp, theta[self.ind_uyuy])

    # def Kdivfx(self, r, rp, theta):
    #     return -self.d0L(r, rp, theta[self.ind_uxux])

    # def Kdivfy(self, r, rp, theta):
    #     return -self.d1L(r, rp, theta[self.ind_uyuy])

    # def Kfyfx(self, r, rp, theta):
    #     return self.d1d0(r, rp, theta[self.ind_pp])

    # def Kdivux(self, r, rp, theta):
    #     return self.d00(r, rp, theta[self.ind_uxux])

    # def Kdivuy(self, r, rp, theta):
    #     return self.d01(r, rp, theta[self.ind_uyuy])

    # ## Kernels include difference of variables
    # def Kuxdifux(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kuxux)(r, rp, theta)

    # def Kuxdifuy(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kuxuy)(r, rp, theta)

    # def Kuydifux(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kuxuy)(r, rp, theta)

    # def Kuydifuy(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kuyuy)(r, rp, theta)

    # def Kuxdifp(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kuxp)(r, rp, theta)

    # def Kuydifp(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kuyp)(r, rp, theta)

    # def Kpdifp(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kpp)(r, rp, theta)

    # def Kfxdifp(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kfxp)(r, rp, theta)

    # def Kfydifp(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kfyp)(r, rp, theta)

    # def Kdivdifp(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kdivp)(r, rp, theta)

    # def Kdifuxdifux(self, r, rp, theta):
    #     return self.setup_kernel_difdif(self.Kuxux)(r, rp, theta)

    # def Kdifuxdifuy(self, r, rp, theta):
    #     return self.setup_kernel_difdif(self.Kuxuy)(r, rp, theta)

    # def Kdifuxfx(self, r, rp, theta):
    #     return self.setup_kernel_include_difference(self.Kuxfx)(r, rp, theta)

    # def Kdifuxfy(self, r, rp, theta):
    #     return self.setup_kernel_include_difference(self.Kuxfy)(r, rp, theta)

    # def Kdifuxdiv(self, r, rp, theta):
    #     return self.setup_kernel_include_difference(self.Kuxdiv)(r, rp, theta)

    # def Kdifuydifuy(self, r, rp, theta):
    #     return self.setup_kernel_difdif(self.Kuyuy)(r, rp, theta)

    # def Kdifuyfx(self, r, rp, theta):
    #     return self.setup_kernel_include_difference(self.Kuyfx)(r, rp, theta)

    # def Kdifuyfy(self, r, rp, theta):
    #     return self.setup_kernel_include_difference(self.Kuyfy)(r, rp, theta)

    # def Kdifuydiv(self, r, rp, theta):
    #     return self.setup_kernel_include_difference(self.Kuydiv)(r, rp, theta)

    # def Kdifuxp(self, r, rp, theta):
    #     return self.setup_kernel_include_difference(self.Kuxp)(r, rp, theta)

    # def Kdifuxdifp(self, r, rp, theta):
    #     return self.setup_kernel_difdif(self.Kuxp)(r, rp, theta)

    # def Kdifuyp(self, r, rp, theta):
    #     return self.setup_kernel_include_difference(self.Kuyp)(r, rp, theta)

    # def Kdifuydifp(self, r, rp, theta):
    #     return self.setup_kernel_difdif(self.Kuyp)(r, rp, theta)

    def Kdifpdifp(self, r, rp, theta):
        return self.setup_kernel_difdif(self.Kpp)(r, rp, theta)

    # def Kfxdifux(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kfxux)(r, rp, theta)

    # def Kfxdifuy(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kfxuy)(r, rp, theta)

    # def Kfydifux(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kfyux)(r, rp, theta)

    # def Kfydifuy(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kfyuy)(r, rp, theta)

    # def Kdivdifux(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kdivux)(r, rp, theta)

    # def Kdivdifuy(self, r, rp, theta):
    #     return self.setup_kernel_include_difference_prime(self.Kdivuy)(r, rp, theta)
