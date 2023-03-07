import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from stopro.GP.kernels import define_kernel


class GPmodel():
    def __init__(self):
        pass

    def cholesky_decompose_non_positive_definite(self, matrix, noise, **kwargs):
        """Compute cholesky decomposition with modifying non positive semedifinite matrix to positive semidifinite matrix"""
        w, v = jnp.linalg.eigh(matrix)
        w = jnp.where(w < 0, 1e-19, w)  # make this pd, psd is insufficient
        matrix_positive_definite = v @ jnp.eye(len(v))*w @ v.T
        matrix_positive_definite_noise = matrix_positive_definite + \
            jnp.diag(noise)

        return jnp.linalg.cholesky(matrix_positive_definite_noise), np.any(w < 0)

    def logpGP(self, δf, Σ, ϵ, approx_non_pd=False):
        """Compute minus log-likelihood of observing δf = f - <f>, for GP with covariance matrix Σ"""
        n = len(δf)
        # jiggle parameter to improve numerical stability of cholesky decomposition
        noise = jnp.ones_like(δf)*ϵ
        if approx_non_pd:
            ####### modify semidifinite to difinite ###############
            L, _is_non_pd = self.cholesky_decompose_non_positive_definite(
                Σ, noise)
        else:
            ######### default ###################
            L = jnp.linalg.cholesky(Σ + jnp.diag(noise))
        v = jnp.linalg.solve(L, δf)
        return (0.5*jnp.dot(v, v) + jnp.sum(jnp.log(jnp.diag(L))) + 0.5*n*jnp.log(2.0*jnp.pi))

    def postGP(self, δfb, Kaa, Kab, Kbb, ϵ, approx_non_pd=False):
        """Compute posterior average and covariance from conditional GP p(fa | xa, xb, fb)
        [fa,fb] ~ 𝒩([μ_fa, μ_fb], [[Kaa, Kab],[Kab^T, Kbb]])])
        fa|fb   ~ 𝒩(μf + Kab Kbb \ (fb - μ_fb) , Kaa - Kab Kbb \ Kab^T)
        """
        noise = jnp.ones(len(Kbb))*ϵ
        if approx_non_pd:
            ################## modify semidifinite to definite ###############
            L, non_pd = self.cholesky_decompose_non_positive_definite(
                Kbb, noise)
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
        V = jnp.array([jnp.linalg.solve(L, c)
                       for c in Kab])  # V = [v_1, v_2, ... ]^t
        Kpost = Kaa - jnp.einsum('ik,jk->ij', V, V)
        return μpost, Kpost  # note should add μ(x*) to average

    def calculate_K_symmetric(self, pts, Ks):
        """Compute symmetric part of covariance matrix (training_K and test_K)
        """
        r_num = len(pts)

        sec = np.zeros(r_num+1, dtype='int')
        r = []
        for i, x in enumerate(pts):
            r.append(x)
            sec[i+1:] += len(x)
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

    def calculate_K_asymmetric(self, train_pts, test_pts, Ks):
        """Compute asymmetric part of covariance matrix (mixed_K)
        """
        rt_num = len(test_pts)
        r_num = len(train_pts)

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
        Σ = jnp.zeros((sect[rt_num], sec[r_num]))
        for i in range(rt_num):
            for j in range(r_num):
                Σ = Σ.at[sect[i]:sect[i+1], sec[j]
                    :sec[j+1]].set(Ks[i][j](rt[i], r[j]))
        return Σ

    def trainingFunction_all(self):
        return NotImplementedError

    def predictingFunction_all(self):
        return NotImplementedError
