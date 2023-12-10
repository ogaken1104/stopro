import jax.numpy as jnp
import numpy as np

from stopro.GP.gp_sinusoidal_without_p_dif import GPufdiv


class GPSinusoidalInferDuxx(GPufdiv):
    def mixedK_all(self, θ, test_pts, train_pts):
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)

        def Kduxxux(r, rp):
            return self.d00(r, rp, θuxux)

        def Kduxxuy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxxfx(r, rp):
            return -self.d0L(r, rp, θuxux)

        def Kduxxfy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxxdiv(r, rp):
            return self.d0d0(r, rp, θuxux)

        Ks = [
            [
                Kduxxux,
                Kduxxuy,
                Kduxxfx,
                Kduxxfy,
                Kduxxdiv,
            ],
        ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, θ, test_pts):
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)

        def Kduxxduxx(r, rp):
            return self.d0d0(r, rp, θuxux)

        Ks = [
            [Kduxxduxx],
        ]

        return self.calculate_K_test(test_pts, Ks)


class GPSinusoidalInferDuyy(GPufdiv):
    def mixedK_all(self, θ, test_pts, train_pts):
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)

        def Kduyyux(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyyuy(r, rp):
            return self.d01(r, rp, θuyuy)

        def Kduyyfx(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyyfy(r, rp):
            return -self.d1L(r, rp, θuyuy)

        def Kduyydiv(r, rp):
            return self.d1d1(r, rp, θuyuy)

        Ks = [
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

        def Kduyyduyy(r, rp):
            return self.d1d1(r, rp, θuyuy)

        Ks = [
            [Kduyyduyy],
        ]

        return self.calculate_K_test(test_pts, Ks)


class GPSinusoidalInferDuxy(GPufdiv):
    def mixedK_all(self, θ, test_pts, train_pts):
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)

        def Kduxyux(r, rp):
            return self.d01(r, rp, θuxux)

        def Kduxyuy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxyfx(r, rp):
            return -self.d1L(r, rp, θuxux)

        def Kduxyfy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxydiv(r, rp):
            return self.d1d0(r, rp, θuxux)

        Ks = [
            [
                Kduxyux,
                Kduxyuy,
                Kduxyfx,
                Kduxyfy,
                Kduxydiv,
            ],
        ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, θ, test_pts):
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)

        def Kduxyduxy(r, rp):
            return self.d1d1(r, rp, θuxux)

        Ks = [
            [Kduxyduxy],
        ]

        return self.calculate_K_test(test_pts, Ks)


class GPSinusoidalInferDuyx(GPufdiv):
    def mixedK_all(self, θ, test_pts, train_pts):
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)

        def Kduyxux(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyxuy(r, rp):
            return self.d00(r, rp, θuyuy)

        def Kduyxfx(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyxfy(r, rp):
            return -self.d0L(r, rp, θuyuy)

        def Kduyxdiv(r, rp):
            return self.d0d1(r, rp, θuyuy)

        Ks = [
            [
                Kduyxux,
                Kduyxuy,
                Kduyxfx,
                Kduyxfy,
                Kduyxdiv,
            ],
        ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, θ, test_pts):
        θuxux, θuyuy, θpp = self.split_hyperparam(theta=θ)

        def Kduyxduyx(r, rp):
            return self.d0d0(r, rp, θuyuy)

        Ks = [
            [Kduyxduyx],
        ]

        return self.calculate_K_test(test_pts, Ks)
