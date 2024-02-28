import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from stopro.GP.gp import GPmodel
from stopro.GP.kernels import define_kernel


class GPSinusoidalGradP(GPmodel):
    def __init__(self, use_gradp_training=False, infer_gradp=False, lbox=None):
        self.use_gradp_training = use_gradp_training
        self.infer_gradp = infer_gradp
        self.lbox = lbox

    def setup_training_and_predicting_functions(self, kernel_type, kernel_form):
        def outermap(f):
            return vmap(vmap(f, in_axes=(None, 0, None)), in_axes=(0, None, None))

        Kernel = define_kernel(kernel_type, kernel_form, lbox=self.lbox)
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

        def trainingK_all(θ, train_pts):
            """
            Args :
            θ  : kernel hyperparameters
            args: training points r_ux,r_uy,r_p,r_fx,r_fy,r_div
            """

            θuxux, θuxuy, θuxpx, θuxpy, θuyuy, θuypx, θuypy, θpxpx,  θpxpy, θpypy = jnp.split(
                θ, 10)

            def Kuxux(r, rp): return K(r, rp, θuxux)
            def Kuxuy(r, rp): return K(r, rp, θuxuy)
            def Kuyuy(r, rp): return K(r, rp, θuyuy)

            def Kuxfx(r, rp): return K(r, rp, θuxpx) - L1(r, rp, θuxux)
            def Kuxfy(r, rp): return K(r, rp, θuxpy) - L1(r, rp, θuxuy)
            def Kuxdiv(r, rp): return d10(r, rp, θuxux)+d11(r, rp, θuxuy)
            def Kuyfx(r, rp): return K(r, rp, θuypx) - L1(r, rp, θuxuy)
            def Kuyfy(r, rp): return K(r, rp, θuypy) - L1(r, rp, θuyuy)

            def Kuydiv(r, rp): return d10_rev(
                r, rp, θuxuy)+d11(r, rp, θuyuy)

            def Kfxfx(r, rp): return K(r, rp, θpxpx) - L1(r, rp,
                                                          θuxpx) - L0(r, rp, θuxpx) + LL(r, rp, θuxux)

            def Kfxfy(r, rp): return K(r, rp, θpxpy) - L1(r, rp,
                                                          θuypx) - L0(r, rp, θuxpy) + LL(r, rp, θuxuy)

            def Kfxdiv(r, rp): return d10_rev(
                r, rp, θuxpx) + d11_rev(r, rp, θuypx) - Ld0(r, rp, θuxux) - Ld1(r, r, θuxuy)

            def Kfyfy(r, rp): return K(r, rp, θpypy) - L1(r, rp,
                                                          θuypy) - L0(r, rp, θuypy) + LL(r, rp, θuyuy)

            def Kfydiv(r, rp): return d10_rev(
                r, rp, θuxpy) + d11_rev(r, rp, θuypy) - Ld0_rev(r, rp, θuxuy) - Ld1(r, r, θuyuy)
            def Kdivdiv(r, rp): return d0d0(r, rp, θuxux)+d0d1(r, rp,
                                                               θuxuy)+d1d0_rev(r, rp, θuxuy)+d1d1(r, rp, θuyuy)

            Ks = [
                [Kuxux, Kuxuy, Kuxfx, Kuxfy, Kuxdiv],
                [Kuyuy, Kuyfx, Kuyfy, Kuydiv],
                [Kfxfx, Kfxfy, Kfxdiv],
                [Kfyfy, Kfydiv],
                [Kdivdiv]
            ]

            return self.calculate_K_symmetric(train_pts, Ks)

        def mixedK_all(θ, test_pts, train_pts):
            θuxux, θuxuy, θuxpx, θuxpy, θuyuy, θuypx, θuypy, θpxpx,  θpxpy, θpypy = jnp.split(
                θ, 10)

            def Kuxux(r, rp): return K(r, rp, θuxux)
            def Kuxuy(r, rp): return K(r, rp, θuxuy)
            def Kuxpx(r, rp): return K(r, rp, θuxpx)
            def Kuxpy(r, rp): return K(r, rp, θuxpy)
            def Kuyuy(r, rp): return K(r, rp, θuyuy)
            def Kuypx(r, rp): return K(r, rp, θuypx)
            def Kuypy(r, rp): return K(r, rp, θuypy)
            def Kpxpx(r, rp): return K(r, rp, θpxpx)
            def Kpxpy(r, rp): return K(r, rp, θpxpy)
            def Kpypy(r, rp): return K(r, rp, θpypy)

            def Kuxfx(r, rp): return K(r, rp, θuxpx) - L1(r, rp, θuxux)
            def Kuxfy(r, rp): return K(r, rp, θuxpy) - L1(r, rp, θuxuy)
            def Kuxdiv(r, rp): return d10(r, rp, θuxux)+d11(r, rp, θuxuy)
            def Kuyfx(r, rp): return K(r, rp, θuypx) - L1(r, rp, θuxuy)
            def Kuyfy(r, rp): return K(r, rp, θuypy) - L1(r, rp, θuyuy)
            def Kuydiv(r, rp): return d10_rev(
                r, rp, θuxuy)+d11(r, rp, θuyuy)

            def Kpxfx(r, rp): return K(r, rp, θpxpx) - L1(r, rp, θuxpx)
            def Kpxfy(r, rp): return K(r, rp, θpxpy) - L1(r, rp, θuypx)
            def Kpxdiv(r, rp): return d10_rev(
                r, rp, θuxpx) + d11_rev(r, rp, θuypx)

            def Kpyfx(r, rp): return K(r, rp, θpxpy) - L1(r, rp, θuxpy)
            def Kpyfy(r, rp): return K(r, rp, θpypy) - L1(r, rp, θuypy)

            def Kpydiv(r, rp): return d10_rev(
                r, rp, θuxpy) + d11_rev(r, rp, θuypy)

            if self.infer_gradp:
                Ks = [
                    [Kuxux, Kuxuy, Kuxfx, Kuxfy, Kuxdiv],
                    [Kuxuy, Kuyuy, Kuyfx, Kuyfy, Kuydiv],
                    [Kuxpx, Kuypx, Kpxfx, Kpxfy, Kpxdiv],
                    [Kuxpy, Kuypy, Kpyfx, Kpyfy, Kpydiv],
                ]
            else:
                Ks = [
                    [Kuxux, Kuxuy, Kuxfx, Kuxfy, Kuxdiv],
                    [Kuxuy, Kuyuy, Kuyfx, Kuyfy, Kuydiv],
                ]

            return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

        def testK_all(θ, test_pts):
            θuxux, θuxuy, θuxpx, θuxpy, θuyuy, θuypx, θuypy, θpxpx,  θpxpy, θpypy = jnp.split(
                θ, 10)

            def Kuxux(r, rp): return K(r, rp, θuxux)
            def Kuxuy(r, rp): return K(r, rp, θuxuy)
            def Kuyuy(r, rp): return K(r, rp, θuyuy)

            if self.infer_gradp:
                def Kuxpx(r, rp): return K(r, rp, θuxpx)
                def Kuxpy(r, rp): return K(r, rp, θuxpy)
                def Kuypx(r, rp): return K(r, rp, θuypx)
                def Kuypy(r, rp): return K(r, rp, θuypy)
                def Kpxpx(r, rp): return K(r, rp, θpxpx)
                def Kpxpy(r, rp): return K(r, rp, θpxpy)
                def Kpypy(r, rp): return K(r, rp, θpypy)
                Ks = [
                    [Kuxux, Kuxuy, Kuxpx, Kuxpy],
                    [Kuyuy, Kuypx, Kuypy],
                    [Kpxpx, Kpxpy],
                    [Kpypy]
                ]
            else:
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
            return self.logpGP(δy, Σ, ϵ)

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
            μposts, Σposts = self.postGP(δfb, Σaa, Σab, Σbb, ϵ)
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

        return jit(trainingFunction_all), jit(predictingFunction_all)
