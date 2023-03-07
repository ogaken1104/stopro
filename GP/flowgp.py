import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from stopro.GP.kernels import define_kernel


@jit
def logpGP(δf, Σ, ϵ):
    """Compute minus log-likelihood of observing δf = f - <f>, for GP with covariance matrix Σ"""
    n = len(δf)
    # jiggle parameter to improve numerical stability of cholesky decomposition
    noise = jnp.ones_like(δf)*ϵ
    L = jnp.linalg.cholesky(Σ + jnp.diag(noise))
#     diffs=Σ-np.dot(L,L.transpose())
#     diff=np.linalg.norm(diffs)
    v = jnp.linalg.solve(L, δf)
    return (0.5*jnp.dot(v, v) + jnp.sum(jnp.log(jnp.diag(L))) + 0.5*n*jnp.log(2.0*jnp.pi))


@jit
def postGP(δfb, Kaa, Kab, Kbb):
    """Compute posterior average and covariance from conditional GP p(fa | xa, xb, fb)
    [fa,fb] ~ 𝒩([μ_fa, μ_fb], [[Kaa, Kab],[Kab^T, Kbb]])])
    fa|fb   ~ 𝒩(μf + Kab Kbb \ (fb - μ_fb) , Kaa - Kab Kbb \ Kab^T)
    """
    L = jnp.linalg.cholesky(Kbb)

    # α = K \ δ f = L^t \ (L | δ f)
    α = jnp.linalg.solve(L.transpose(), jnp.linalg.solve(L, δfb))

    # μpost - μ(x*) = Kab Kbb \ δf(x) = Kab . α
    μpost = jnp.dot(Kab, α)

    # Kpost = Kaa - Kab Kbb | Kab^T
    #       = Kaa - W
    # W_ij  = v_i . v_j
    # v_i   = (L | c_i) ; c_i the i-th column of Kba, i-th row of Kab
    V = jnp.array([jnp.linalg.solve(L, c)
                   for c in Kab])  # V = [v_1, v_2, ... ]^t
    Kpost = Kaa - jnp.einsum('ik,jk->ij', V, V)
    return μpost, Kpost  # note should add μ(x*) to average


def flowgp_2D(kernel_type, kernel_form):
    def outermap(f):
        return vmap(vmap(f, in_axes=(None, 0, None)), in_axes=(0, None, None))

    Kernel = define_kernel(kernel_type, kernel_form)
    # define kernel for use
    K = jit(outermap(Kernel))
    def Kernel_rev(r1, r2, θ): return Kernel(r2, r1, θ)
    K_rev = jit(outermap(Kernel_rev))

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

    def _LL(r, rp, θ): return jnp.sum(jnp.diag(jax.hessian(_L1, 0)(r, rp, θ)))

    def _dij(i, j, r, rp, θ): return grad(Kernel, i)(r, rp, θ)[j]
    def _d10(r, rp, θ): return _dij(1, 0, r, rp, θ)
    def _d11(r, rp, θ): return _dij(1, 1, r, rp, θ)
    def _d10(r, rp, θ): return grad(Kernel, 1)(r, rp, θ)[0]
    def _d11(r, rp, θ): return grad(Kernel, 1)(r, rp, θ)[1]
    L0 = jit(outermap(_L0))
    L1 = jit(outermap(_L1))
    d0d0 = jit(outermap(_d0d0))
    d0d1 = jit(outermap(_d0d1))
    d1d0 = jit(outermap(_d1d0))
    d1d1 = jit(outermap(_d1d1))
    d0L = jit(outermap(_d0L))
    d1L = jit(outermap(_d1L))
    Ld0 = jit(outermap(_Ld0))
    Ld1 = jit(outermap(_Ld1))
    LL = jit(outermap(_LL))
    d10 = jit(outermap(_d10))
    d11 = jit(outermap(_d11))

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
    L0_rev = jit(outermap(_L0_rev))
    L1_rev = jit(outermap(_L1_rev))
    d0d0_rev = jit(outermap(_d0d0_rev))
    d0d1_rev = jit(outermap(_d0d1_rev))
    d1d0_rev = jit(outermap(_d1d0_rev))
    d1d1_rev = jit(outermap(_d1d1_rev))
    d0L_rev = jit(outermap(_d0L_rev))
    d1L_rev = jit(outermap(_d1L_rev))
    Ld0_rev = jit(outermap(_Ld0_rev))
    Ld1_rev = jit(outermap(_Ld1_rev))
    LL_rev = jit(outermap(_LL_rev))
    d10_rev = jit(outermap(_d10_rev))
    d11_rev = jit(outermap(_d11_rev))

    @jit
    def trainingK_all(θ, train_pts):
        """
        Args :
          θ  : kernel hyperparameters
         args: training points r_ux,r_uy,r_p,r_fx,r_fy,r_div
        """
        r_num = len(train_pts)
    #     print(θ,r_num)
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, r_num)
        sec = np.zeros(r_num+1, dtype='int')
        r = []
        for i, x in enumerate(train_pts):
            r.append(x)
            sec[i+1:] += len(x)
        # K_normal??
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

        Ks = [
            [Kuxux, Kuxuy, Kuxp, Kuxfx, Kuxfy, Kuxdiv],
            [Kuyuy, Kuyp, Kuyfx, Kuyfy, Kuydiv],
            [Kpp,  Kpfx,  Kpfy,  Kpdiv],
            [Kfxfx, Kfxfy, Kfxdiv],
            [Kfyfy, Kfydiv],
            [Kdivdiv]
        ]

        Σ = jnp.zeros((sec[r_num], sec[r_num]))
        for i in range(r_num):
            for j in range(i, r_num):
                # upper triangular matrix
                Σ = Σ.at[sec[i]:sec[i+1], sec[j]:sec[j+1]
                         ].set(Ks[i][j-i](r[i], r[j]))
                if not j == i:
                    # transpose
                    Σ = Σ.at[sec[j]:sec[j+1], sec[i]:sec[i+1]
                             ].set(jnp.transpose(Σ[sec[i]:sec[i+1], sec[j]:sec[j+1]]))
        return Σ

    @jit
    def mixedK_all(θ, test_pts, train_pts):
        rt_num = len(test_pts)
        r_num = len(train_pts)
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, r_num)

        rt = []
        sect = np.zeros(rt_num+1, dtype='int')
        for i, x in enumerate(test_pts):
            rt.append(x)
            sect[i+1:] += len(x)

        r = []
        sec = np.zeros(r_num+1, dtype='int')
        for i, x in enumerate(train_pts):
            r.append(x)
            sec[i+1:] += len(x)

        def Kuxux(r, rp): return K(r, rp, θuxux)
        def Kuxuy(r, rp): return K(r, rp, θuxuy)
        def Kuyuy(r, rp): return K(r, rp, θuyuy)
        def Kuxp(r, rp): return K(r, rp, θuxp)
        def Kuyp(r, rp): return K(r, rp, θuyp)
        def Kpp(r, rp): return K(r, rp, θpp)
        def Kuxfx(r, rp): return d10(r, rp, θuxp)-L1(r, rp, θuxux)
        def Kuyfx(r, rp): return d10(r, rp, θuyp)-L1_rev(r, rp, θuxuy)
        def Kpfx(r, rp): return d10(r, rp, θpp)-L1_rev(r, rp, θuxp)

        def Kuyux(r, rp): return K_rev(r, rp, θuxuy)
        def Kpux(r, rp): return K_rev(r, rp, θuxp)
        def Kpuy(r, rp): return K_rev(r, rp, θuyp)

        def Kuxfy(r, rp): return d11(r, rp, θuxp)-L1(r, rp, θuxuy)
        def Kuyfy(r, rp): return d11(r, rp, θuyp)-L1(r, rp, θuyuy)
        def Kpfy(r, rp): return d11(r, rp, θpp)-L1_rev(r, rp, θuyp)

        def Kuxdiv(r, rp): return d10(r, rp, θuxux)+d11(r, rp, θuxuy)
        def Kuydiv(r, rp): return d10_rev(r, rp, θuxuy)+d11(r, rp, θuyuy)
        def Kpdiv(r, rp): return d10_rev(r, rp, θuxp)+d11_rev(r, rp, θuyp)

        Ks = [
            [Kuxux, Kuxuy, Kuxp, Kuxfx, Kuxfy, Kuxdiv],
            [Kuyux, Kuyuy, Kuyp, Kuyfx, Kuyfy, Kuydiv],
            [Kpux,  Kpuy,  Kpp,  Kpfx,  Kpfy,  Kpdiv]
        ]
        Σ = jnp.zeros((sect[rt_num], sec[r_num]))
        for i in range(rt_num):
            for j in range(r_num):
                Σ = Σ.at[sect[i]:sect[i+1], sec[j]:sec[j+1]].set(Ks[i][j](rt[i], r[j]))
        return Σ

    @jit
    def testK_all(θ, r_test):
        rt_num = len(r_test)
        θuxux, θuyuy, θpp, θuxuy, θuxp, θuyp = jnp.split(θ, 6)
        rt = r_test
        sect = np.zeros(rt_num+1, dtype='int')
        for i, x in enumerate(r_test):
            sect[i+1:] += len(x)

        def Kuxux(r, rp): return K(r, rp, θuxux)
        def Kuxuy(r, rp): return K(r, rp, θuxuy)
        def Kuyuy(r, rp): return K(r, rp, θuyuy)
        def Kuxp(r, rp): return K(r, rp, θuxp)
        def Kuyp(r, rp): return K(r, rp, θuyp)
        def Kpp(r, rp): return K(r, rp, θpp)
        Σ = jnp.zeros((sect[-1], sect[-1]))
        Ks = [
            [Kuxux, Kuxuy, Kuxp],
            [Kuyuy, Kuyp],
            [Kpp]
        ]
        for i in range(rt_num):
            for j in range(i, rt_num):
                # upper triangular matrix
                Σ = Σ.at[sect[i]:sect[i+1], sect[j]:sect[j+1]].set(Ks[i][j-i](rt[i], rt[j]))
                if not j == i:
                    # transpose
                    Σ = Σ.at[sect[j]:sect[j+1], sect[i]:sect[i+1]
                             ].set(jnp.transpose(Σ[sect[i]:sect[i+1], sect[j]:sect[j+1]]))
        return Σ

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
        return logpGP(δy, Σ, ϵ)

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
        Σbb = trainingK_all(θ, r_train) + jnp.diag(jnp.ones(nb)*ϵ)
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
        μposts, Σposts = postGP(δfb, Σaa, Σab, Σbb)
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
