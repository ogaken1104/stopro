import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from stopro.GP.gp import GPmodel
from stopro.GP.kernels import define_kernel


class GPPeriodic(GPmodel):
    def __init__(self, lbox_x=None, lbox_y=None, use_difp=False, use_difu=False):
        self.lbox_x = lbox_x
        self.lbox_y = lbox_y

    def setup_training_and_predicting_functions(self, kernel_type, kernel_form, approx_non_pd=False):
        def outermap(f):
            return vmap(vmap(f, in_axes=(None, 0, None)), in_axes=(0, None, None))

        Kernel = define_kernel(kernel_type, kernel_form)
        # define kernel for use
        K = outermap(Kernel)
        def Kernel_rev(r1, r2, θ): return Kernel(r2, r1, θ)
        K_rev = outermap(Kernel_rev)

        # define operators
        def _L0(r, rp, θ): return jnp.sum(
            jnp.diag(jax.hessian(Kernel, 0)(r, rp, θ)))

        def _L1(r, rp, θ): return jnp.sum(
            jnp.diag(jax.hessian(Kernel, 1)(r, rp, θ)))

        def _d0d0(r, rp, θ): return jax.hessian(
            Kernel, [0, 1])(r, rp, θ)[0][1][0, 0]

        def _d0d1(r, rp, θ): return jax.hessian(
            Kernel, [0, 1])(r, rp, θ)[0][1][0, 1]

        def _d1d0(r, rp, θ): return jax.hessian(
            Kernel, [0, 1])(r, rp, θ)[0][1][1, 0]
        def _d1d1(r, rp, θ): return jax.hessian(
            Kernel, [0, 1])(r, rp, θ)[0][1][1, 1]

        def _d0L(r, rp, θ): return grad(_L1, 0)(r, rp, θ)[0]
        def _d1L(r, rp, θ): return grad(_L1, 0)(r, rp, θ)[1]

        def _Ld0(r, rp, θ): return jnp.sum(
            jnp.diag(jax.hessian(_d10, 0)(r, rp, θ)))

        def _Ld1(r, rp, θ): return jnp.sum(
            jnp.diag(jax.hessian(_d11, 0)(r, rp, θ)))

        def _LL(r, rp, θ): return jnp.sum(
            jnp.diag(jax.hessian(_L1, 0)(r, rp, θ)))

        def _dij(i, j, r, rp, θ): return grad(Kernel, i)(r, rp, θ)[j]
        def _d10(r, rp, θ): return _dij(1, 0, r, rp, θ)
        def _d11(r, rp, θ): return _dij(1, 1, r, rp, θ)
        def _d10(r, rp, θ): return grad(Kernel, 1)(r, rp, θ)[0]
        def _d11(r, rp, θ): return grad(Kernel, 1)(r, rp, θ)[1]
        def _d00(r, rp, θ): return grad(Kernel, 0)(r, rp, θ)[0]
        def _d01(r, rp, θ): return grad(Kernel, 0)(r, rp, θ)[1]
        L0 = outermap(_L0)
        L1 = outermap(_L1)
        d0d0 = outermap(_d0d0)
        d0d1 = outermap(_d0d1)
        d1d0 = outermap(_d1d0)
        d1d1 = outermap(_d1d1)
        d0L = outermap(_d0L)
        d1L = outermap(_d1L)
        Ld0 = outermap(_Ld0)
        Ld1 = outermap(_Ld1)
        LL = outermap(_LL)
        d10 = outermap(_d10)
        d11 = outermap(_d11)
        d00 = outermap(_d00)
        d01 = outermap(_d01)

        # define operators for rev
        def _L0_rev(r, rp, θ): return jnp.sum(
            jnp.diag(jax.hessian(Kernel_rev, 0)(r, rp, θ)))

        def _L1_rev(r, rp, θ): return jnp.sum(
            jnp.diag(jax.hessian(Kernel_rev, 1)(r, rp, θ)))

        def _d0d0_rev(r, rp, θ): return jax.hessian(
            Kernel_rev, [0, 1])(r, rp, θ)[0][1][0, 0]

        def _d0d1_rev(r, rp, θ): return jax.hessian(
            Kernel_rev, [0, 1])(r, rp, θ)[0][1][0, 1]

        def _d1d0_rev(r, rp, θ): return jax.hessian(
            Kernel_rev, [0, 1])(r, rp, θ)[0][1][1, 0]
        def _d1d1_rev(r, rp, θ): return jax.hessian(
            Kernel_rev, [0, 1])(r, rp, θ)[0][1][1, 1]

        def _d0L_rev(r, rp, θ): return grad(_L1_rev, 0)(r, rp, θ)[0]
        def _d1L_rev(r, rp, θ): return grad(_L1_rev, 0)(r, rp, θ)[1]

        def _Ld0_rev(r, rp, θ): return jnp.sum(
            jnp.diag(jax.hessian(_d10_rev, 0)(r, rp, θ)))

        def _Ld1_rev(r, rp, θ): return jnp.sum(
            jnp.diag(jax.hessian(_d11_rev, 0)(r, rp, θ)))
        def _LL_rev(r, rp, θ): return jnp.sum(
            jnp.diag(jax.hessian(_L1_rev, 0)(r, rp, θ)))

        def _dij_rev(i, j, r, rp, θ): return grad(Kernel_rev, i)(r, rp, θ)[j]
        def _d10_rev(r, rp, θ): return _dij_rev(1, 0, r, rp, θ)
        def _d11_rev(r, rp, θ): return _dij_rev(1, 1, r, rp, θ)
        def _d10_rev(r, rp, θ): return grad(Kernel_rev, 1)(r, rp, θ)[0]
        def _d11_rev(r, rp, θ): return grad(Kernel_rev, 1)(r, rp, θ)[1]
        L0_rev = outermap(_L0_rev)
        L1_rev = outermap(_L1_rev)
        d0d0_rev = outermap(_d0d0_rev)
        d0d1_rev = outermap(_d0d1_rev)
        d1d0_rev = outermap(_d1d0_rev)
        d1d1_rev = outermap(_d1d1_rev)
        d0L_rev = outermap(_d0L_rev)
        d1L_rev = outermap(_d1L_rev)
        Ld0_rev = outermap(_Ld0_rev)
        Ld1_rev = outermap(_Ld1_rev)
        LL_rev = outermap(_LL_rev)
        d10_rev = outermap(_d10_rev)
        d11_rev = outermap(_d11_rev)

        def setup_kernel_include_dif_prime(K_func, lbox):
            def K_difp(r, rp): return K_func(
                r, rp+lbox) - K_func(r, rp)
            return K_difp

        def setup_kernel_include_difu(K_func, lbox):
            def K_difu(r, rp): return K_func(
                r+lbox, rp) - K_func(r, rp)
            return K_difu
        # def setup_kernel_inputs_switched(K_func):
        #     def K_switched(rp, r): return K_func(r, rp)
        #     return K_switched

        def setup_kernel_difdif(K_func, lbox1, lbox2):
            def K_difdif(r, rp):
                return K_func(r+lbox1, rp+lbox2)-K_func(r+lbox1, rp)-K_func(r, rp+lbox2)+K_func(r, rp)
            return K_difdif

        def trainingK_all(θ, train_pts):
            """
            Args :
            θ  : kernel hyperparameters
            args: training points r_ux,r_uy,r_p,r_fx,r_fy,r_div
            """

            θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

            def Kuxux(r, rp): return K(r, rp, θuxux)
            def Kuxuy(r, rp): return K(r, rp, θuxuy)
            def Kuyuy(r, rp): return K(r, rp, θuyuy)
            def Kuxp(r, rp): return K(r, rp, θuxp)
            def Kuyp(r, rp): return K(r, rp, θuyp)
            def Kpp(r, rp): return K(r, rp, θpp)
            def Kuxfx(r, rp): return d10(r, rp, θuxp)-L1(r, rp, θuxux)
            def Kuyfx(r, rp): return d10(r, rp, θuyp)-L1_rev(r, rp, θuxuy)
            def Kpfx(r, rp): return d10(r, rp, θpp)-L1_rev(r, rp, θuxp)
            def Kfxfx(r, rp): return d0d0(r, rp, θpp) - \
                d0L_rev(r, rp, θuxp)-Ld0(r, rp, θuxp)+LL(r, rp, θuxux)

            def Kuxfy(r, rp): return d11(r, rp, θuxp)-L1(r, rp, θuxuy)
            def Kuyfy(r, rp): return d11(r, rp, θuyp)-L1(r, rp, θuyuy)
            def Kpfy(r, rp): return d11(r, rp, θpp)-L1_rev(r, rp, θuyp)

            def Kfxfy(r, rp): return d0d1(r, rp, θpp) - \
                d0L_rev(r, rp, θuyp)-Ld1(r, rp, θuxp)+LL(r, rp, θuxuy)
            def Kfyfy(r, rp): return d1d1(r, rp, θpp) - \
                d1L_rev(r, rp, θuyp)-Ld1(r, rp, θuyp)+LL(r, rp, θuyuy)

            def Kuxdiv(r, rp): return d10(r, rp, θuxux)+d11(r, rp, θuxuy)
            def Kuydiv(r, rp): return d10_rev(r, rp, θuxuy)+d11(r, rp, θuyuy)
            def Kpdiv(r, rp): return d10_rev(r, rp, θuxp)+d11_rev(r, rp, θuyp)

            def Kfxdiv(r, rp): return d0d0_rev(r, rp, θuxp) + \
                d0d1_rev(r, rp, θuyp)-Ld0(r, rp, θuxux)-Ld1(r, rp, θuxuy)

            def Kfydiv(r, rp): return d1d0_rev(r, rp, θuxp)+d1d1_rev(r,
                                                                     rp, θuyp)-Ld0_rev(r, rp, θuxuy)-Ld1(r, rp, θuyuy)

            def Kdivdiv(r, rp): return d0d0(r, rp, θuxux)+d0d1(r, rp,
                                                               θuxuy)+d1d0_rev(r, rp, θuxuy)+d1d1(r, rp, θuyuy)

            Kuxdifux = setup_kernel_include_dif_prime(Kuxux)
            Kuxdifuy = setup_kernel_include_dif_prime(Kuxuy)
            Kuydifux = setup_kernel_include_dif_prime(Kuxuy)
            Kuydifuy = setup_kernel_include_dif_prime(Kuyuy)
            Kdifuxdifux = setup_kernel_difdif(Kuxux)
            Kdifuxdifuy = setup_kernel_difdif(Kuxuy)
            Kdifuxfx = setup_kernel_include_difu(Kuxfx)
            Kdifuxfy = setup_kernel_include_difu(Kuxfy)
            Kdifuxdiv = setup_kernel_include_difu(Kuxdiv)
            Kdifuydifuy = setup_kernel_difdif(Kuyuy)
            Kdifuyfx = setup_kernel_include_difu(Kuyfx)
            Kdifuyfy = setup_kernel_include_difu(Kuyfy)
            Kdifuydiv = setup_kernel_include_difu(Kuydiv)

            Kdifuxp = setup_kernel_include_difu(Kuxp)
            Kdifuxdifp = setup_kernel_difdif(Kuxp)
            Kdifuyp = setup_kernel_include_difu(Kuyp)
            Kdifuydifp = setup_kernel_difdif(Kuyp)

            Kuxdifux_x = setup_kernel_include_dif_prime(Kuxux, self.lbox_x)
            Kuxdifuy_x = setup_kernel_include_dif_prime(Kuxuy, self.lbox_x)
            Kuydifux_x = setup_kernel_include_dif_prime(Kuxuy, self.lbox_x)
            Kuydifuy_x = setup_kernel_include_dif_prime(Kuyuy, self.lbox_x)
            Kuxdifux_y = setup_kernel_include_dif_prime(Kuxux, self.lbox_y)
            Kuxdifuy_y = setup_kernel_include_dif_prime(Kuxuy, self.lbox_y)
            Kuydifux_y = setup_kernel_include_dif_prime(Kuxuy, self.lbox_y)
            Kuydifuy_y = setup_kernel_include_dif_prime(Kuyuy, self.lbox_y)

            Kdifux_xdifux_x = setup_kernel_difdif(
                Kuxux, self.lbox_x, self.lbox_x)
            Kdifux_xdifux_y = setup_kernel_difdif(
                Kuxux, self.lbox_x, self.lbox_y)
            Kdifux_xdifuy_x = setup_kernel_difdif(
                Kuxuy, self.lbox_x, self.lbox_x)
            Kdifux_xdifuy_y = setup_kernel_difdif(
                Kuxuy, self.lbox_x, self.lbox_y)
            Kdifux_ydifux_y = setup_kernel_difdif(
                Kuxux, self.lbox_y, self.lbox_y)
            Kdifux_ydifuy_x = setup_kernel_difdif(
                Kuxuy, self.lbox_y, self.lbox_x)
            Kdifux_ydifuy_y = setup_kernel_difdif(
                Kuxuy, self.lbox_y, self.lbox_y)
            Kdifuy_xdifuy_x = setup_kernel_difdif(
                Kuyuy, self.lbox_x, self.lbox_x)
            Kdifuy_xdifuy_y = setup_kernel_difdif(
                Kuyuy, self.lbox_x, self.lbox_y)
            Kdifuy_ydifuy_y = setup_kernel_difdif(
                Kuyuy, self.lbox_y, self.lbox_y)

            Kdifux_xfx = setup_kernel_include_difu(Kuxfx, self.lbox_x)
            Kdifux_xfy = setup_kernel_include_difu(Kuxfy, self.lbox_x)
            Kdifux_xdiv = setup_kernel_include_difu(Kuxdiv, self.lbox_x)
            Kdifux_yfx = setup_kernel_include_difu(Kuxfx, self.lbox_y)
            Kdifux_yfy = setup_kernel_include_difu(Kuxfy, self.lbox_y)
            Kdifux_ydiv = setup_kernel_include_difu(Kuxdiv, self.lbox_y)

            Kdifuy_xfx = setup_kernel_include_difu(Kuyfx, self.lbox_x)
            Kdifuy_xfy = setup_kernel_include_difu(Kuyfy, self.lbox_x)
            Kdifuy_xdiv = setup_kernel_include_difu(Kuydiv, self.lbox_x)
            Kdifuy_yfx = setup_kernel_include_difu(Kuyfx, self.lbox_y)
            Kdifuy_yfy = setup_kernel_include_difu(Kuyfy, self.lbox_y)
            Kdifuy_ydiv = setup_kernel_include_difu(Kuydiv, self.lbox_y)

            def Kfxp(r, rp): return d00(r, rp, θpp)-L0(r, rp, θuxp)
            def Kfyp(r, rp): return d01(r, rp, θpp)-L0(r, rp, θuyp)
            def Kdivp(r, rp): return d00(r, rp, θuxp) + d01(r, rp, θuyp)

            Kuxdifp_x = setup_kernel_include_dif_prime(Kuxp, self.lbox_x)
            Kuydifp_x = setup_kernel_include_dif_prime(Kuyp, self.lbox_x)
            Kfxdifp_x = setup_kernel_include_dif_prime(Kfxp, self.lbox_x)
            Kfydifp_x = setup_kernel_include_dif_prime(Kfyp, self.lbox_x)
            Kdivdifp_x = setup_kernel_include_dif_prime(Kdivp, self.lbox_x)
            Kuxdifp_y = setup_kernel_include_dif_prime(Kuxp, self.lbox_y)
            Kuydifp_y = setup_kernel_include_dif_prime(Kuyp, self.lbox_y)
            Kfxdifp_y = setup_kernel_include_dif_prime(Kfxp, self.lbox_y)
            Kfydifp_y = setup_kernel_include_dif_prime(Kfyp, self.lbox_y)
            Kdivdifp_y = setup_kernel_include_dif_prime(Kdivp, self.lbox_y)

            Kdifux_xdifp_x = setup_kernel_difdif(
                Kuxp, self.lbox_x, self.lbox_x)
            Kdifux_xdifp_y = setup_kernel_difdif(
                Kuxp, self.lbox_x, self.lbox_y)
            Kdifuy_xdifp_x = setup_kernel_difdif(
                Kuyp, self.lbox_x, self.lbox_x)
            Kdifuy_xdifp_y = setup_kernel_difdif(
                Kuyp, self.lbox_x, self.lbox_y)
            Kdifux_ydifp_x = setup_kernel_difdif(
                Kuxp, self.lbox_y, self.lbox_x)
            Kdifux_ydifp_y = setup_kernel_difdif(
                Kuxp, self.lbox_y, self.lbox_y)
            Kdifuy_ydifp_x = setup_kernel_difdif(
                Kuyp, self.lbox_y, self.lbox_x)
            Kdifuy_ydifp_y = setup_kernel_difdif(
                Kuyp, self.lbox_y, self.lbox_y)

            Kdifp_xdifp_x = setup_kernel_difdif(Kpp, self.lbox_x, self.lbox_x)
            Kdifp_xdifp_y = setup_kernel_difdif(Kpp, self.lbox_x, self.lbox_y)
            Kdifp_ydifp_y = setup_kernel_difdif(Kpp, self.lbox_y, self.lbox_y)

            Ks = [
                [Kuxux, Kuxuy, Kuxdifux_x, Kuxdifux_y, Kuxdifuy_x, Kuxdifuy_y,
                    Kuxfx, Kuxfy, Kuxdiv, Kuxdifp_x, Kuxdifp_y],
                [Kuyuy, Kuydifux_x, Kuydifux_y, Kuydifuy_x, Kuydifuy_y,
                    Kuyfx, Kuyfy, Kuydiv, Kuydifp_x, Kuydifp_y],
                [Kdifux_xdifux_x, Kdifux_xdifux_y, Kdifux_xdifuy_x, Kdifux_xdifuy_y, Kdifux_xfx,
                    Kdifux_xfy, Kdifux_xdiv, Kdifux_xdifp_x, Kdifux_xdifp_y],
                [Kdifux_ydifux_y, Kdifux_ydifuy_x, Kdifux_ydifuy_y, Kdifux_yfx,
                    Kdifux_yfy, Kdifux_ydiv, Kdifux_ydifp_x, Kdifux_ydifp_y],
                [Kdifuy_xdifuy_x, Kdifuy_xdifuy_y, Kdifuy_xfx,
                    Kdifuy_xfy, Kdifuy_xdiv, Kdifuy_xdifp_x, Kdifuy_xdifp_y],
                [Kdifuy_ydifuy_y, Kdifuy_yfx,
                    Kdifuy_yfy, Kdifuy_ydiv, Kdifuy_ydifp_x, Kdifuy_ydifp_y],
                [Kfxfx, Kfxfy, Kfxdiv, Kfxdifp_x, Kfxdifp_y],
                [Kfyfy, Kfydiv, Kfydifp_x, Kfydifp_y],
                [Kdivdiv, Kdivdifp_x, Kdivdifp_y],
                [Kdifp_xdifp_x, Kdifp_xdifp_y],
                [Kdifp_ydifp_y]
            ]

            return self.calculate_K_symmetric(train_pts, Ks)

        def mixedK_all(θ, test_pts, train_pts):
            θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

            def Kuxux(r, rp): return K(r, rp, θuxux)
            def Kuxuy(r, rp): return K(r, rp, θuxuy)
            def Kuyuy(r, rp): return K(r, rp, θuyuy)
            def Kuxp(r, rp): return K(r, rp, θuxp)
            def Kuyp(r, rp): return K(r, rp, θuyp)
            def Kpp(r, rp): return K(r, rp, θpp)
            def Kuxfx(r, rp): return d10(r, rp, θuxp)-L1(r, rp, θuxux)
            def Kuyfx(r, rp): return d10(r, rp, θuyp)-L1_rev(r, rp, θuxuy)
            def Kpfx(r, rp): return d10(r, rp, θpp)-L1_rev(r, rp, θuxp)
            def Kdivp(r, rp): return d00(r, rp, θuxp) + d01(r, rp, θuyp)

            def Kuyux(r, rp): return K_rev(r, rp, θuxuy)
            def Kpux(r, rp): return K_rev(r, rp, θuxp)
            def Kpuy(r, rp): return K_rev(r, rp, θuyp)

            def Kuxfy(r, rp): return d11(r, rp, θuxp)-L1(r, rp, θuxuy)
            def Kuyfy(r, rp): return d11(r, rp, θuyp)-L1(r, rp, θuyuy)
            def Kpfy(r, rp): return d11(r, rp, θpp)-L1_rev(r, rp, θuyp)

            def Kuxdiv(r, rp): return d10(r, rp, θuxux)+d11(r, rp, θuxuy)
            def Kuydiv(r, rp): return d10_rev(r, rp, θuxuy)+d11(r, rp, θuyuy)
            def Kpdiv(r, rp): return d10_rev(r, rp, θuxp)+d11_rev(r, rp, θuyp)

            Kuxdifux_x = setup_kernel_include_dif_prime(Kuxux, self.lbox_x)
            Kuxdifp_y = setup_kernel_include_dif_prime(Kuxp, self.lbox_y)
            Kuxdifp_x = setup_kernel_include_dif_prime(Kuxp, self.lbox_x)
            Kuxdifuy_y = setup_kernel_include_dif_prime(Kuxuy, self.lbox_y)
            Kuxdifux_y = setup_kernel_include_dif_prime(Kuxux, self.lbox_y)
            Kuxdifuy_x = setup_kernel_include_dif_prime(Kuxuy, self.lbox_x)
            Kuxdifux_x = setup_kernel_include_dif_prime(Kuxux, self.lbox_x)

            Kuydifux_x = setup_kernel_include_dif_prime(Kuyux, self.lbox_x)
            Kuydifp_y = setup_kernel_include_dif_prime(Kuyp, self.lbox_y)
            Kuydifp_x = setup_kernel_include_dif_prime(Kuyp, self.lbox_x)
            Kuydifuy_y = setup_kernel_include_dif_prime(Kuyuy, self.lbox_y)
            Kuydifux_y = setup_kernel_include_dif_prime(Kuyux, self.lbox_y)
            Kuydifuy_x = setup_kernel_include_dif_prime(Kuyuy, self.lbox_x)
            Kuydifux_x = setup_kernel_include_dif_prime(Kuyux, self.lbox_x)

            Ks = [
                [Kuxux, Kuxuy, Kuxdifux_x, Kuxdifux_y, Kuxdifuy_x, Kuxdifuy_y,
                    Kuxfx, Kuxfy, Kuxdiv, Kuxdifp_x, Kuxdifp_y],
                [Kuyuy, Kuydifux_x, Kuydifux_y, Kuydifuy_x, Kuydifuy_y,
                    Kuyfx, Kuyfy, Kuydiv, Kuydifp_x, Kuydifp_y],
            ]

            return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

        def testK_all(θ, test_pts):
            θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)

            def Kuxux(r, rp): return K(r, rp, θuxux)
            def Kuxuy(r, rp): return K(r, rp, θuxuy)
            def Kuyuy(r, rp): return K(r, rp, θuyuy)
            def Kuxp(r, rp): return K(r, rp, θuxp)
            def Kuyp(r, rp): return K(r, rp, θuyp)
            def Kpp(r, rp): return K(r, rp, θpp)

            Ks = [
                [Kuxux, Kuxuy],
                [Kuyuy],
            ]

            return self.calculate_K_symmetric(test_pts, Ks)

        def trainingFunction_all(θ, *args):
            """Returns minus log-likelihood given Kernel hyperparamters θ and training data args
            args = velocity position, velocity average, velocity values, 
                force position, force average, force values, 
                jiggle parameter
            """
            # r,μ,f,ϵ=args
            r, μ, f, ϵ = args
            r_num = len(r)
            for i in range(r_num):
                if i == 0:
                    δy = jnp.array(f[i]-μ[i])
                else:
                    δy = jnp.concatenate([δy, f[i]-μ[i]], 0)
            Σ = trainingK_all(θ, r)
            return self.logpGP(δy, Σ, ϵ, approx_non_pd=approx_non_pd)

        def predictingFunction_all(θ, *args):
            """Returns conditional posterior average and covariance matrix given Kernel hyperparamters θ  and test and training data
            args = test velocity position, test velocity average,
                training velocity position, training velocity average, training velocity values
                training force position, training force average, training force values
                jiggle parameter

            Returns
            -----------------
            μpost=[μux,μuy,μp]
            Σpost=[Σux,Σuy,Σp]
            """
            r_test, μ_test, r_train, μ, f_train, ϵ = args
            nb = 0
            for r in r_train:
                nb += len(r)
            Σbb = trainingK_all(θ, r_train)
            Σab = mixedK_all(θ, r_test, r_train)
            Σaa = testK_all(θ, r_test)
            for i in range(len(r_train)):
                if i == 0:
                    δfb = jnp.array(f_train[i]-μ[i])
                else:
                    δfb = jnp.concatenate([δfb, f_train[i]-μ[i]])
                    # create single training array, with velocities and forces (second derivatives)
    #         print(f'δy={δy}')
    #         print(f'Σ={Σ}')
            μposts, Σposts = self.postGP(
                δfb, Σaa, Σab, Σbb, ϵ, approx_non_pd=approx_non_pd)
            # seperate μpost,Σpost to 3 section (ux,uy,p)
            sec0 = 0
            sec1 = 0
            μpost = []
            Σpost = []
            for i in range(len(r_test)):
                sec1 += len(r_test[i])
                μpost.append(μposts[sec0:sec1])
                Σpost.append(Σposts[sec0:sec1, sec0:sec1])
                sec0 += len(r_test[i])
                # 一応解決ちょっと疑問残る
                μpost[i] += μ_test[i]
            return μpost, Σpost

        return trainingFunction_all, predictingFunction_all
        # return jit(trainingFunction_all), jit(predictingFunction_all)
