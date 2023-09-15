import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap


class GPmodel:
    """
    The base class for any Gaussian Process model
    """

    def __init__(self, approx_non_pd: bool = False):
        """
        Args:
            approx_non_pd: if approximate non positive semi-difinite covariance matrix with a poisitive semi-difinite matix
        """
        self.approx_non_pd = approx_non_pd

    def cholesky_decompose_non_positive_definite(self, matrix, noise, **kwargs):
        """Compute cholesky decomposition with modifying non positive semedifinite matrix to positive semidifinite matrix"""
        w, v = jnp.linalg.eigh(matrix)
        w = jnp.where(w < 0, 1e-19, w)  # make this pd, psd is insufficient
        matrix_positive_definite = v @ jnp.eye(len(v)) * w @ v.T
        matrix_positive_definite_noise = matrix_positive_definite + jnp.diag(noise)

        return jnp.linalg.cholesky(matrix_positive_definite_noise), np.any(w < 0)

    def logpGP(self, δf, Σ, ϵ, approx_non_pd=False):
        """Compute minus log-likelihood of observing δf = f - <f>, for GP with covariance matrix Σ"""
        n = len(δf)
        # jiggle parameter to improve numerical stability of cholesky decomposition
        noise = jnp.ones_like(δf) * ϵ
        if approx_non_pd:
            ####### modify semidifinite to difinite ###############
            L, _is_non_pd = self.cholesky_decompose_non_positive_definite(Σ, noise)
        else:
            ######### default ###################
            L = jnp.linalg.cholesky(Σ + jnp.diag(noise))
        v = jnp.linalg.solve(L, δf)
        return (
            0.5 * jnp.dot(v, v)
            + jnp.sum(jnp.log(jnp.diag(L)))
            + 0.5 * n * jnp.log(2.0 * jnp.pi)
        )

    def postGP(self, δfb, Kaa, Kab, Kbb, ϵ, approx_non_pd=False):
        """Compute posterior average and covariance from conditional GP p(fa | xa, xb, fb)
        [fa,fb] ~ 𝒩([μ_fa, μ_fb], [[Kaa, Kab],[Kab^T, Kbb]])])
        fa|fb   ~ 𝒩(μf + Kab Kbb \ (fb - μ_fb) , Kaa - Kab Kbb \ Kab^T)
        """
        noise = jnp.ones(len(Kbb)) * ϵ
        if approx_non_pd:
            ################## modify semidifinite to definite ###############
            L, non_pd = self.cholesky_decompose_non_positive_definite(Kbb, noise)
        else:
            ################# default ######################
            L = jnp.linalg.cholesky(Kbb + jnp.diag(noise))

        # α = K \ δ f = L^t \ (L | δ f)
        α = jnp.linalg.solve(L.transpose(), jnp.linalg.solve(L, δfb))

        # μpost - μ(x*) = Kab Kbb \ δf(x) = Kab . α
        μpost = jnp.dot(Kab, α)

        # Kpost = Kaa - Kab Kbb | Kab^T
        #       = Kaa - W
        # W_ij  = v_i . v_j
        # v_i   = (L | c_i) ; c_i the i-th column of Kba, i-th row of Kab
        V = jnp.array([jnp.linalg.solve(L, c) for c in Kab])  # V = [v_1, v_2, ... ]^t
        Kpost = Kaa - jnp.einsum("ik,jk->ij", V, V)
        return μpost, Kpost  # note should add μ(x*) to average

    def calculate_K_symmetric(self, pts, Ks, std_noise_list=[None] * 8):
        """Compute symmetric part of covariance matrix (training_K and test_K)"""
        r_num = len(pts)

        sec = np.zeros(r_num + 1, dtype="int")
        r = []
        for i, x in enumerate(pts):
            r.append(x)
            sec[i + 1 :] += len(x)
        Σ = jnp.zeros((sec[r_num], sec[r_num]))
        for i in range(r_num):
            for j in range(i, r_num):
                # upper triangular matrix
                Σ = Σ.at[sec[i] : sec[i + 1], sec[j] : sec[j + 1]].set(
                    Ks[i][j - i](r[i], r[j])
                )
                # if i == j and std_noise_list[i] is not None:
                #     var_noise_list = jnp.full(len(r[i]), std_noise_list[i]**2)
                #     var_noise_matrix = jnp.diag(var_noise_list)
                #     Σ = Σ.at[sec[i]:sec[i+1], sec[j]:sec[j+1]
                #             ].add(var_noise_matrix)
                if not j == i:
                    # transpose
                    Σ = Σ.at[sec[j] : sec[j + 1], sec[i] : sec[i + 1]].set(
                        jnp.transpose(Σ[sec[i] : sec[i + 1], sec[j] : sec[j + 1]])
                    )
        return Σ

    def calculate_K_asymmetric(self, train_pts, test_pts, Ks):
        """Compute asymmetric part of covariance matrix (mixed_K)"""
        rt_num = len(test_pts)
        r_num = len(train_pts)

        rt = []
        sect = np.zeros(rt_num + 1, dtype="int")
        for i, x in enumerate(test_pts):
            rt.append(x)
            sect[i + 1 :] += len(x)

        r = []
        sec = np.zeros(r_num + 1, dtype="int")
        for i, x in enumerate(train_pts):
            r.append(x)
            sec[i + 1 :] += len(x)
        Σ = jnp.zeros((sect[rt_num], sec[r_num]))
        for i in range(rt_num):
            for j in range(r_num):
                Σ = Σ.at[sect[i] : sect[i + 1], sec[j] : sec[j + 1]].set(
                    Ks[i][j](rt[i], r[j])
                )
        return Σ

    def trainingFunction_all(self, θ, *args):
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
                δy = jnp.array(f[i] - μ[i])
            else:
                δy = jnp.concatenate([δy, f[i] - μ[i]], 0)
        Σ = self.trainingK_all(θ, r)
        return self.logpGP(δy, Σ, ϵ, approx_non_pd=self.approx_non_pd)

    def predictingFunction_all(self, θ, *args):
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
        Σbb = self.trainingK_all(θ, r_train)
        Σab = self.mixedK_all(θ, r_test, r_train)
        Σaa = self.testK_all(θ, r_test)
        for i in range(len(r_train)):
            if i == 0:
                δfb = jnp.array(f_train[i] - μ[i])
            else:
                δfb = jnp.concatenate([δfb, f_train[i] - μ[i]])
                # create single training array, with velocities and forces (second derivatives)
        μposts, Σposts = self.postGP(
            δfb, Σaa, Σab, Σbb, ϵ, approx_non_pd=self.approx_non_pd
        )
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
            μpost[i] += μ_test[i]
        return μpost, Σpost

    def trainingK_all(self):
        raise NotImplementedError

    def mixedK_all(self):
        raise NotImplementedError

    def testK_all(self):
        raise NotImplementedError
