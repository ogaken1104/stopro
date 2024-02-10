import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from stopro.GP.gp_sinusoidal_independent import GPSinusoidalWithoutPIndependent


class GPSinusoidalInferDifP(GPSinusoidalWithoutPIndependent):
    def Kpux(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kpuy(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)

    def Kpfx(self, r, rp, theta):
        return self.d10(r, rp, theta[self.ind_pp])

    def Kpfy(self, r, rp, theta):
        return self.d11(r, rp, theta[self.ind_pp])

    def Kpdiv(self, r, rp, theta):
        return self.Kzero(r, rp, self.dummy_theta)
    
    def Kdifpux(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kpux)(r, rp, theta)

    def Kdifpuy(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kpuy)(r, rp, theta)

    def Kdifpfx(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kpfx)(r, rp, theta)

    def Kdifpfy(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kpfy)(r, rp, theta)

    def Kdifpdiv(self, r, rp, theta):
        return self.setup_kernel_include_difference(self.Kpdiv)(r, rp, theta)

    def Kdifpdifux(self, r, rp, theta):
        return self.setup_kernel_difdif(self.Kpux)(r, rp, theta)

    def Kdifpdifuy(self, r, rp, theta):
        return self.setup_kernel_difdif(self.Kpuy)(r, rp, theta)
    
    def setup_trainingKs(self):
        """
        Args :
        """
        self.trainingKs = [
            [self.Kuxux, self.Kuxuy, self.Kuxdifux, self.Kuxdifuy, self.Kuxfx, self.Kuxfy, self.Kuxdiv],
            [self.Kuyuy, self.Kuydifux, self.Kuydifuy, self.Kuyfx, self.Kuyfy, self.Kuydiv],
            [
                self.Kdifuxdifux,
                self.Kdifuxdifuy,
                self.Kdifuxfx,
                self.Kdifuxfy,
                self.Kdifuxdiv,
            ],
            [self.Kdifuydifuy, self.Kdifuyfx, self.Kdifuyfy, self.Kdifuydiv],
            [self.Kfxfx, self.Kfxfy, self.Kfxdiv],
            [self.Kfyfy, self.Kfydiv],
            [self.Kdivdiv],
        ]

    def setup_mixedKs(self):
        self.mixedKs = [
            [self.Kdifpux, self.Kdifpuy, self.Kdifpdifux, self.Kdifpdifuy, self.Kdifpfx, self.Kdifpfy, self.Kdifpdiv],
        ]

    def setup_testKs(self):
        self.testKs = [
            [self.Kdifpdifp],
        ]

class GPSinusoidalInferUWithoutDifP(GPSinusoidalWithoutPIndependent):
    def setup_mixedKs(self):
        self.mixedKs = [
            [self.Kuxux, self.Kuxuy, self.Kuxdifux, self.Kuxdifuy, self.Kuxfx, self.Kuxfy, self.Kuxdiv],
            [self.Kuyux, self.Kuyuy, self.Kuydifux, self.Kuydifuy, self.Kuyfx, self.Kuyfy, self.Kuydiv],
        ]

    def setup_testKs(self):
        self.testKs = [
            [self.Kuxux, self.Kuxuy],
            [self.Kuyuy],
        ]

class GPSinusoidalInferGovWithoutDifP(GPSinusoidalWithoutPIndependent):
    def setup_mixedKs(self):
        self.mixedKs = [
            [self.Kfxux, self.Kfxuy, self.Kfxdifux, self.Kfxdifuy, self.Kfxfx, self.Kfxfy, self.Kfxdiv],
            [self.Kfyux, self.Kfyuy, self.Kfydifux, self.Kfydifuy, self.Kfyfx, self.Kfyfy, self.Kfydiv],
            [self.Kdivux, self.Kdivuy, self.Kdivdifux, self.Kdivdifuy, self.Kdivfx, self.Kdivfy, self.Kdivdiv],
        ]

    def setup_testKs(self):
        self.testKs = [
            [self.Kfxfx, self.Kfxuy, self.Kfxdiv],
            [self.Kfyfy, self.Kfydiv],
            [self.Kdivdiv],
        ]
