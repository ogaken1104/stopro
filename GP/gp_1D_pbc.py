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


def gp_1D_pbc(model_param, lbox):
    kernel_type = model_param['kernel_type']
    kernel_form = model_param['kernel_form']
    distance_func = model_param['distance_func']
    def outermap(f): return vmap(
        vmap(f, in_axes=(None, 0, None)), in_axes=(0, None, None))

    Kernel = define_kernel(kernel_type, kernel_form,
                           input_dim=1, distance_func=distance_func)
    # define kernel for use
    # define operators
    def L(f, i): return grad(grad(f, i), i)  # L = d^2 / dx^2
    _L1K = L(Kernel, 1)                     # (L_2 K)(x*,x')
    _L0K = L(Kernel, 0)
    _LLK = L(_L1K, 0)

    K = jit(outermap(Kernel))
    L0K = jit(outermap(_L0K))
    L1K = jit(outermap(_L1K))
    LLK = jit(outermap(_LLK))

    @jit
    def trainingK_all(θ, train_pts):
        """
        Args :
          θ  : kernel hyperparameters
         args: training points r_ux,r_uy,r_p,r_fx,r_fy,r_div
        """
        r_num = len(train_pts)
    #     print(θ,r_num)
        θyy = θ
        sec = np.zeros(r_num+1, dtype='int')
        r = []
        for i, x in enumerate(train_pts):
            r.append(x)
            sec[i+1:] += len(x)

        def Kyy(r, rp): return K(r, rp, θyy)
        def Kyly(r, rp): return L1K(r, rp, θyy)
        def Klyly(r, rp): return LLK(r, rp, θyy)
        def Kypbcy(r, rp): return K(r, rp+lbox, θyy) - K(r, rp, θyy)
        def Klypbcy(r, rp): return L0K(r, rp+lbox, θyy) - L0K(r, rp, θyy)
        def Kpbcypbcy(r, rp): return K(r+lbox, rp+lbox, θyy) - \
            K(r+lbox, rp, θyy) - K(r, rp+lbox, θyy) + K(r, rp, θyy)

        Ks = [
            [Kyy, Kyly, Kypbcy],
            [Klyly, Klypbcy],
            [Kpbcypbcy]
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
        θyy = θ

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

        def Kyy(r, rp): return K(r, rp, θyy)
        def Kyly(r, rp): return L1K(r, rp, θyy)
        def Kypbcy(r, rp): return K(r, rp+lbox, θyy) - K(r, rp, θyy)

        Ks = [
            [Kyy, Kyly, Kypbcy]
        ]
        Σ = jnp.zeros((sect[rt_num], sec[r_num]))
        for i in range(rt_num):
            for j in range(r_num):
                Σ = Σ.at[sect[i]:sect[i+1], sec[j]
                    :sec[j+1]].set(Ks[i][j](rt[i], r[j]))
        return Σ

    @jit
    def testK_all(θ, r_test):
        rt_num = len(r_test)
        θyy = θ
        rt = r_test
        sect = np.zeros(rt_num+1, dtype='int')
        for i, x in enumerate(r_test):
            sect[i+1:] += len(x)

        def Kyy(r, rp): return K(r, rp, θyy)
        Σ = jnp.zeros((sect[-1], sect[-1]))
        Ks = [
            [Kyy],
        ]
        for i in range(rt_num):
            for j in range(i, rt_num):
                # upper triangular matrix
                Σ = Σ.at[sect[i]:sect[i+1], sect[j]
                    :sect[j+1]].set(Ks[i][j-i](rt[i], rt[j]))
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
