import jax.numpy as jnp
import numpy as np

from stopro.GP.gp_sinusoidal_without_p_dif import GPufdiv, GPufdivNoisyu


class GPSinusoidalInferDu(GPufdiv):
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


class GPSinusoidalInferDuP(GPufdiv):
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

        def Kpux(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kpuy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kpfx(r, rp):
            return self.d10(r, rp, θpp)

        def Kpfy(r, rp):
            return self.d11(r, rp, θpp)

        def Kpdiv(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

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
            [
                Kpux,
                Kpuy,
                Kpfx,
                Kpfy,
                Kpdiv,
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

        def Kduxxp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxyp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyxp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyyp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kpp(r, rp):
            return self.K(r, rp, θpp)

        Ks = [
            [Kduxxduxx, Kduxxduxy, Kduxxduyx, Kduxxduyy, Kduxxp],
            [Kduxyduxy, Kduxyduyx, Kduxyduyy, Kduxyp],
            [Kduyxduyx, Kduyxduyy, Kduyxp],
            [Kduyyduyy, Kduyyp],
            [Kpp],
        ]

        return self.calculate_K_test(test_pts, Ks)


class GPufdivNoisyuInferDuP(GPufdivNoisyu):
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

        def Kpux(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kpuy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kpfx(r, rp):
            return self.d10(r, rp, θpp)

        def Kpfy(r, rp):
            return self.d11(r, rp, θpp)

        def Kpdiv(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        Ks = [
            [
                Kduxxux,
                Kduxxuy,
                Kduxxux,
                Kduxxuy,
                Kduxxfx,
                Kduxxfy,
                Kduxxdiv,
            ],
            [
                Kduxyux,
                Kduxyuy,
                Kduxyux,
                Kduxyuy,
                Kduxyfx,
                Kduxyfy,
                Kduxydiv,
            ],
            [
                Kduyxux,
                Kduyxuy,
                Kduyxux,
                Kduyxuy,
                Kduyxfx,
                Kduyxfy,
                Kduyxdiv,
            ],
            [
                Kduyyux,
                Kduyyuy,
                Kduyyux,
                Kduyyuy,
                Kduyyfx,
                Kduyyfy,
                Kduyydiv,
            ],
            [
                Kpux,
                Kpuy,
                Kpux,
                Kpuy,
                Kpfx,
                Kpfy,
                Kpdiv,
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

        def Kduxxp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxyp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyxp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyyp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kpp(r, rp):
            return self.K(r, rp, θpp)

        Ks = [
            [Kduxxduxx, Kduxxduxy, Kduxxduyx, Kduxxduyy, Kduxxp],
            [Kduxyduxy, Kduxyduyx, Kduxyduyy, Kduxyp],
            [Kduyxduyx, Kduyxduyy, Kduyxp],
            [Kduyyduyy, Kduyyp],
            [Kpp],
        ]

        return self.calculate_K_test(test_pts, Ks)
