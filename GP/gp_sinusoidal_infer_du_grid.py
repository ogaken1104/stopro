import jax.numpy as jnp
import numpy as np

from stopro.GP.gp_sinusoidal_without_p import GPSinusoidalWithoutP


class GPSinusoidalInferDuxx(GPSinusoidalWithoutP):
    def mixedK_all(self, θ, test_pts, train_pts):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

        def Kduxxux(r, rp):
            return self.d00(r, rp, θuxux)

        def Kduxxuy(r, rp):
            return self.d00(r, rp, θuxuy)

        Kduxxdifux = self.setup_kernel_include_difference_prime(Kduxxux)

        Kduxxdifuy = self.setup_kernel_include_difference_prime(Kduxxuy)

        def Kduxxfx(r, rp):
            return self.d0d0(r, rp, θuxp) - self.d0L(r, rp, θuxux)

        def Kduxxfy(r, rp):
            return self.d0d1(r, rp, θuxp) - self.d0L(r, rp, θuxuy)

        def Kduxxdiv(r, rp):
            return self.d0d0(r, rp, θuxux) + self.d0d1(r, rp, θuxuy)

        def Kduxxp(r, rp):
            return self.d00(r, rp, θuxp)

        Kduxxdifp = self.setup_kernel_include_difference_prime(Kduxxp)

        Ks = [
            [
                Kduxxux,
                Kduxxuy,
                Kduxxdifux,
                Kduxxdifuy,
                Kduxxfx,
                Kduxxfy,
                Kduxxdiv,
                Kduxxdifp,
            ],
        ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, θ, test_pts):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

        def Kduxxduxx(r, rp):
            return self.d0d0(r, rp, θuxux)

        Ks = [
            [Kduxxduxx],
        ]

        return self.calculate_K_symmetric(test_pts, Ks)


class GPSinusoidalInferDuyy(GPSinusoidalWithoutP):
    def mixedK_all(self, θ, test_pts, train_pts):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

        def Kduyyux(r, rp):
            return self.d01_rev(r, rp, θuxuy)

        def Kduyyuy(r, rp):
            return self.d01(r, rp, θuyuy)

        Kduyydifux = self.setup_kernel_include_difference_prime(Kduyyux)

        Kduyydifuy = self.setup_kernel_include_difference_prime(Kduyyuy)

        def Kduyyfx(r, rp):
            return self.d1d0(r, rp, θuyp) - self.d1L_rev(r, rp, θuxuy)

        def Kduyyfy(r, rp):
            return self.d1d1(r, rp, θuyp) - self.d1L(r, rp, θuyuy)

        def Kduyydiv(r, rp):
            return self.d1d0_rev(r, rp, θuxuy) + self.d1d1(r, rp, θuyuy)

        def Kduyyp(r, rp):
            return self.d01(r, rp, θuyp)

        Kduyydifp = self.setup_kernel_include_difference_prime(Kduyyp)

        Ks = [
            [
                Kduyyux,
                Kduyyuy,
                Kduyydifux,
                Kduyydifuy,
                Kduyyfx,
                Kduyyfy,
                Kduyydiv,
                Kduyydifp,
            ],
        ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, θ, test_pts):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

        def Kduyyduyy(r, rp):
            return self.d1d1(r, rp, θuyuy)

        Ks = [
            [Kduyyduyy],
        ]

        return self.calculate_K_symmetric(test_pts, Ks)
