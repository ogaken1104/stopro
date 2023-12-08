import jax.numpy as jnp
import numpy as np

from stopro.GP.gp_sinusoidal_without_p_dif import GPSinusoidalWithoutPDif


class GPSinusoidalInferDu(GPSinusoidalWithoutPDif):
    def mixedK_all(self, θ, test_pts, train_pts):
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)

        def Kduxxux(r, rp):
            return self.d00(r, rp, θuxux)

        def Kduxyux(r, rp):
            return self.d01(r, rp, θuxux)

        def Kduyxux(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyyux(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxxuy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxyuy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyxuy(r, rp):
            return self.d00(r, rp, θuyuy)

        def Kduyyuy(r, rp):
            return self.d01(r, rp, θuyuy)

        def Kduxxfx(r, rp):
            return -self.d0L(r, rp, θuxux)

        def Kduxyfx(r, rp):
            return -self.d1L(r, rp, θuxux)

        def Kduyxfx(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyyfx(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxxfy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxyfy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyxfy(r, rp):
            return -self.d0L(r, rp, θuyuy)

        def Kduyyfy(r, rp):
            return -self.d1L(r, rp, θuyuy)

        def Kduxxdiv(r, rp):
            return self.d0d0(r, rp, θuxux)

        def Kduxydiv(r, rp):
            return self.d1d0(r, rp, θuxux)

        def Kduyxdiv(r, rp):
            return self.d0d1(r, rp, θuyuy)

        def Kduyydiv(r, rp):
            return self.d1d1(r, rp, θuyuy)

        Ks = [
            [
                Kduxxux,
                Kduxxuy,
                Kduxxfx,
                Kduxxfy,
                Kduxxdiv,
            ],
            [
                Kduxyux,
                Kduxyuy,
                Kduxyfx,
                Kduxyfy,
                Kduxydiv,
            ],
            [
                Kduyxux,
                Kduyxuy,
                Kduyxfx,
                Kduyxfy,
                Kduyxdiv,
            ],
            [
                Kduyyux,
                Kduyyuy,
                Kduyyfx,
                Kduyyfy,
                Kduyydiv,
            ],
        ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, θ, test_pts):
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)

        def Kduxxduxx(r, rp):
            return self.d0d0(r, rp, θuxux)

        def Kduxxduxy(r, rp):
            return self.d0d1(r, rp, θuxux)

        def Kduxxduyx(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxxduyy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxyduxy(r, rp):
            return self.d1d1(r, rp, θuxux)

        def Kduxyduyx(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxyduyy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyxduyx(r, rp):
            return self.d0d0(r, rp, θuyuy)

        def Kduyxduyy(r, rp):
            return self.d0d1(r, rp, θuyuy)

        def Kduyyduyy(r, rp):
            return self.d1d1(r, rp, θuyuy)

        Ks = [
            [Kduxxduxx, Kduxxduxy, Kduxxduyx, Kduxxduyy],
            [Kduxyduxy, Kduxyduyx, Kduxyduyy],
            [Kduyxduyx, Kduyxduyy],
            [Kduyyduyy],
        ]

        return self.calculate_K_test(test_pts, Ks)
