import jax.numpy as jnp
import numpy as np

from stopro.GP.gp_sinusoidal_without_p import GPSinusoidalWithoutP


class GPSinusoidalInferDu(GPSinusoidalWithoutP):
    def mixedK_all(self, θ, test_pts, train_pts):
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

        def Kduxxux(r, rp):
            return self.d00(r, rp, θuxux)

        def Kduxyux(r, rp):
            return self.d01(r, rp, θuxux)

        def Kduyxux(r, rp):
            return self.d00_rev(r, rp, θuxuy)

        def Kduyyux(r, rp):
            return self.d01_rev(r, rp, θuxuy)

        def Kduxxuy(r, rp):
            return self.d00(r, rp, θuxuy)

        def Kduxyuy(r, rp):
            return self.d01(r, rp, θuxuy)

        def Kduyxuy(r, rp):
            return self.d00(r, rp, θuyuy)

        def Kduyyuy(r, rp):
            return self.d01(r, rp, θuyuy)

        Kduxxdifux = self.setup_kernel_include_difference_prime(Kduxxux)
        Kduxydifux = self.setup_kernel_include_difference_prime(Kduxyux)
        Kduyxdifux = self.setup_kernel_include_difference_prime(Kduyxux)
        Kduyydifux = self.setup_kernel_include_difference_prime(Kduyyux)

        Kduxxdifuy = self.setup_kernel_include_difference_prime(Kduxxuy)
        Kduxydifuy = self.setup_kernel_include_difference_prime(Kduxyuy)
        Kduyxdifuy = self.setup_kernel_include_difference_prime(Kduyxuy)
        Kduyydifuy = self.setup_kernel_include_difference_prime(Kduyyuy)

        def Kduxxfx(r, rp):
            return self.d0d0(r, rp, θuxp) - self.d0L(r, rp, θuxux)

        def Kduxyfx(r, rp):
            return self.d1d0(r, rp, θuxp) - self.d1L(r, rp, θuxux)

        def Kduyxfx(r, rp):
            return self.d0d0(r, rp, θuyp) - self.d0L_rev(r, rp, θuxuy)

        def Kduyyfx(r, rp):
            return self.d1d0(r, rp, θuyp) - self.d1L_rev(r, rp, θuxuy)

        def Kduxxfy(r, rp):
            return self.d0d1(r, rp, θuxp) - self.d0L(r, rp, θuxuy)

        def Kduxyfy(r, rp):
            return self.d1d1(r, rp, θuxp) - self.d1L(r, rp, θuxuy)

        def Kduyxfy(r, rp):
            return self.d0d1(r, rp, θuyp) - self.d0L(r, rp, θuyuy)

        def Kduyyfy(r, rp):
            return self.d1d1(r, rp, θuyp) - self.d1L(r, rp, θuyuy)

        def Kduxxdiv(r, rp):
            return self.d0d0(r, rp, θuxux) + self.d0d1(r, rp, θuxuy)

        def Kduxydiv(r, rp):
            return self.d1d0(r, rp, θuxux) + self.d1d1(r, rp, θuxuy)

        def Kduyxdiv(r, rp):
            return self.d0d0_rev(r, rp, θuxuy) + self.d0d1(r, rp, θuyuy)

        def Kduyydiv(r, rp):
            return self.d1d0_rev(r, rp, θuxuy) + self.d1d1(r, rp, θuyuy)

        def Kduxxp(r, rp):
            return self.d00(r, rp, θuxp)

        def Kduxyp(r, rp):
            return self.d01(r, rp, θuxp)

        def Kduyxp(r, rp):
            return self.d00(r, rp, θuyp)

        def Kduyyp(r, rp):
            return self.d01(r, rp, θuyp)

        Kduxxdifp = self.setup_kernel_include_difference_prime(Kduxxp)
        Kduxydifp = self.setup_kernel_include_difference_prime(Kduxyp)
        Kduyxdifp = self.setup_kernel_include_difference_prime(Kduyxp)
        Kduyydifp = self.setup_kernel_include_difference_prime(Kduyyp)

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
            [
                Kduxyux,
                Kduxyuy,
                Kduxydifux,
                Kduxydifuy,
                Kduxyfx,
                Kduxyfy,
                Kduxydiv,
                Kduxydifp,
            ],
            [
                Kduyxux,
                Kduyxuy,
                Kduyxdifux,
                Kduyxdifuy,
                Kduyxfx,
                Kduyxfy,
                Kduyxdiv,
                Kduyxdifp,
            ],
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

        def Kduxxduxx(r, rp):
            return self.d0d0(r, rp, θuxux)

        def Kduxxduxy(r, rp):
            return self.d0d1(r, rp, θuxux)

        def Kduxxduyx(r, rp):
            return self.d0d0(r, rp, θuxuy)

        def Kduxxduyy(r, rp):
            return self.d0d1(r, rp, θuxuy)

        def Kduxyduxy(r, rp):
            return self.d1d1(r, rp, θuxux)

        def Kduxyduyx(r, rp):
            return self.d1d0(r, rp, θuxuy)

        def Kduxyduyy(r, rp):
            return self.d1d1(r, rp, θuxuy)

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

        return self.calculate_K_symmetric(test_pts, Ks)
