from stopro.GP.gp_sinusoidal_independent import GPSinusoidalWithoutPIndependent


class GPSinusoidalInferDuDirect(GPSinusoidalWithoutPIndependent):
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

        Kduxxdifux = self.setup_kernel_include_difference_prime(Kduxxux)
        Kduxydifux = self.setup_kernel_include_difference_prime(Kduxyux)
        Kduyxdifux = self.setup_kernel_include_difference_prime(Kduyxux)
        Kduyydifux = self.setup_kernel_include_difference_prime(Kduyyux)

        Kduxxdifuy = self.setup_kernel_include_difference_prime(Kduxxuy)
        Kduxydifuy = self.setup_kernel_include_difference_prime(Kduxyuy)
        Kduyxdifuy = self.setup_kernel_include_difference_prime(Kduyxuy)
        Kduyydifuy = self.setup_kernel_include_difference_prime(Kduyyuy)

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

        def Kduxxp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduxyp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyxp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kduyyp(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        Kduxxdifp = self.setup_kernel_include_difference_prime(Kduxxp)
        Kduxydifp = self.setup_kernel_include_difference_prime(Kduxyp)
        Kduyxdifp = self.setup_kernel_include_difference_prime(Kduyxp)
        Kduyydifp = self.setup_kernel_include_difference_prime(Kduyyp)

        def Kpux(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kpuy(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        Kpdifux = self.setup_kernel_include_difference_prime(Kpux)
        Kpdifuy = self.setup_kernel_include_difference_prime(Kpuy)

        def Kpfx(r, rp):
            return self.d10(r, rp, θpp)

        def Kpfy(r, rp):
            return self.d11(r, rp, θpp)

        def Kpdiv(r, rp):
            return self.Kzero(r, rp, self.dummy_theta)

        def Kpp(r, rp):
            return self.K(r, rp, θpp)

        Kpdifp = self.setup_kernel_include_difference_prime(Kpp)

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
            [
                Kpux,
                Kpuy,
                Kpdifux,
                Kpdifuy,
                Kpfx,
                Kpfy,
                Kpdiv,
                Kpdifp,
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
