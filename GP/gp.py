import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap


class GPmodel:
    """
    The base class for any Gaussian Process model
    """

    def __init__(self, index_optimize_noise: jnp.array = None):
        """
        Args:
            approx_non_pd: if approximate non positive semi-difinite covariance matrix with a poisitive semi-difinite matix
        """
        self.index_optimize_noise = False

    def _add_jiggle(self, Σ, jiggle_constant, noise_parameter=None):
        jiggle = jnp.ones(len(Σ)) * jiggle_constant
        return Σ + jnp.diag(jiggle)

    def _add_jiggle_noise(self, Σ, jiggle_constant, noise_parameter=None):
        # noise_jiggle = jnp.ones(len(Σ)) * jiggle_constant
        noise_jiggle = jnp.ones(len(Σ))
        r_index_noise_start = self.index_optimize_noise[0]
        r_index_noise_end = self.index_optimize_noise[-1]
        index_noise_start = self.sec_tr[r_index_noise_start]
        index_noise_end = self.sec_tr[r_index_noise_end + 1]
        # noise = jnp.ones(index_noise_end - index_noise_start) * noise_parameter
        noise = jnp.ones(index_noise_end - index_noise_start)
        noise_jiggle = noise_jiggle.at[index_noise_start:index_noise_end].multiply(
            jnp.exp(noise_parameter)
        )
        noise_jiggle = noise_jiggle.at[index_noise_end:].multiply(jiggle_constant)
        return Σ + jnp.diag(noise_jiggle)

    def logpGP(self, δf, Σ, ϵ):
        """Compute minus log-likelihood of observing δf = f - <f>, for GP with covariance matrix Σ"""
        n = len(δf)
        # jiggle parameter to improve numerical stability of cholesky decomposition
        # noise = jnp.ones_like(δf) * ϵ
        # if approx_non_pd:
        #     ####### modify semidifinite to difinite ###############
        #     L, _is_non_pd = self.cholesky_decompose_non_positive_definite(Σ, noise)
        # else:
        #     ######### default ###################
        # L = jnp.linalg.cholesky(Σ + jnp.diag(noise))
        L = jnp.linalg.cholesky(Σ)
        v = jnp.linalg.solve(L, δf)
        return (
            0.5 * jnp.dot(v, v)
            + jnp.sum(jnp.log(jnp.diag(L)))
            + 0.5 * n * jnp.log(2.0 * jnp.pi)
        )

    def postGP(self, δfb, Kaa, Kab, Kbb, ϵ):
        """Compute posterior average and covariance from conditional GP p(fa | xa, xb, fb)
        [fa,fb] ~ 𝒩([μ_fa, μ_fb], [[Kaa, Kab],[Kab^T, Kbb]])])
        fa|fb   ~ 𝒩(μf + Kab Kbb \ (fb - μ_fb) , Kaa - Kab Kbb \ Kab^T)
        """
        # noise = jnp.ones(len(Kbb)) * ϵ
        # if approx_non_pd:
        #     ################## modify semidifinite to definite ###############
        #     L, non_pd = self.cholesky_decompose_non_positive_definite(Kbb, noise)
        # else:
        #     ################# default ######################
        # L = jnp.linalg.cholesky(Kbb + jnp.diag(noise))
        L = jnp.linalg.cholesky(Kbb)

        # α = K \ δ f = L^t \ (L | δ f)
        α = jnp.linalg.solve(L.transpose(), jnp.linalg.solve(L, δfb))

        # μpost - μ(x*) = Kab Kbb \ δf(x) = Kab . α
        μpost = jnp.dot(Kab, α)

        # Kpost = Kaa - Kab Kbb | Kab^T
        #       = Kaa - W
        # W_ij  = v_i . v_j
        # v_i   = (L | c_i) ; c_i the i-th column of Kba, i-th row of Kab
        ### TODO we should eliminate for loop here
        V = jnp.array([jnp.linalg.solve(L, c) for c in Kab])  # V = [v_1, v_2, ... ]^t
        Kpost = Kaa - jnp.einsum("ik,jk->ij", V, V)
        return μpost, Kpost  # note should add μ(x*) to average

    def calculate_K_training(self, pts, Ks, std_noise_list=[None] * 8):
        """Compute symmetric part of covariance matrix (training_K and test_K)"""
        Σ = jnp.zeros((self.sec_tr[self.num_tr], self.sec_tr[self.num_tr]))
        for i in range(self.num_tr):
            for j in range(i, self.num_tr):
                # upper triangular matrix
                Σ = Σ.at[
                    self.sec_tr[i] : self.sec_tr[i + 1],
                    self.sec_tr[j] : self.sec_tr[j + 1],
                ].set(Ks[i][j - i](pts[i], pts[j]))
                if not j == i:
                    # transpose
                    Σ = Σ.at[
                        self.sec_tr[j] : self.sec_tr[j + 1],
                        self.sec_tr[i] : self.sec_tr[i + 1],
                    ].set(
                        jnp.transpose(
                            Σ[
                                self.sec_tr[i] : self.sec_tr[i + 1],
                                self.sec_tr[j] : self.sec_tr[j + 1],
                            ]
                        )
                    )
        return Σ

    def calculate_K_test(self, pts, Ks, std_noise_list=[None] * 8):
        """Compute symmetric part of covariance matrix (training_K and test_K)"""
        Σ = jnp.zeros((self.sec_te[self.num_te], self.sec_te[self.num_te]))
        for i in range(self.num_te):
            for j in range(i, self.num_te):
                # upper triangular matrix
                Σ = Σ.at[
                    self.sec_te[i] : self.sec_te[i + 1],
                    self.sec_te[j] : self.sec_te[j + 1],
                ].set(Ks[i][j - i](pts[i], pts[j]))
                if not j == i:
                    # transpose
                    Σ = Σ.at[
                        self.sec_te[j] : self.sec_te[j + 1],
                        self.sec_te[i] : self.sec_te[i + 1],
                    ].set(
                        jnp.transpose(
                            Σ[
                                self.sec_te[i] : self.sec_te[i + 1],
                                self.sec_te[j] : self.sec_te[j + 1],
                            ]
                        )
                    )
        return Σ

    def calculate_K_asymmetric(self, train_pts, test_pts, Ks):
        """Compute asymmetric part of covariance matrix (mixed_K)"""
        Σ = jnp.zeros((self.sec_te[self.num_te], self.sec_tr[self.num_tr]))
        for i in range(self.num_te):
            for j in range(self.num_tr):
                Σ = Σ.at[
                    self.sec_te[i] : self.sec_te[i + 1],
                    self.sec_tr[j] : self.sec_tr[j + 1],
                ].set(Ks[i][j](test_pts[i], train_pts[j]))
        return Σ

    def trainingFunction_all(self, theta, *args):
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
        θ, noise = self.split_hyp_and_noise(theta)
        Σ = self.trainingK_all(θ, r)
        Σ = self.add_eps_to_sigma(Σ, ϵ, noise_parameter=noise)
        return self.logpGP(δy, Σ, ϵ)

    def predictingFunction_all(self, theta, *args):
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
        θ, noise = self.split_hyp_and_noise(theta)
        Σbb = self.trainingK_all(θ, r_train)
        Σab = self.mixedK_all(θ, r_test, r_train)
        Σbb = self.add_eps_to_sigma(Σbb, ϵ, noise_parameter=noise)
        Σaa = self.testK_all(θ, r_test)
        for i in range(len(r_train)):
            if i == 0:
                δfb = jnp.array(f_train[i] - μ[i])
            else:
                δfb = jnp.concatenate([δfb, f_train[i] - μ[i]])
                # create single training array, with velocities and forces (second derivatives)
        μposts, Σposts = self.postGP(δfb, Σaa, Σab, Σbb, ϵ)
        # seperate μpost,Σpost to 3 self.sec_teion (ux,uy,p)
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

    def calc_sec(self, pts):
        sec = np.concatenate([np.zeros(1, dtype=int), np.cumsum([len(x) for x in pts])])
        return sec

    def set_constants(self, *args, only_training=False):
        if only_training:
            r_train, μ, f_train, ϵ = args
        else:
            r_test, μ_test, r_train, μ, f_train, ϵ = args
            self.num_te = len(r_test)
            self.sec_te = self.calc_sec(r_test)
        self.num_tr = len(r_train)
        self.sec_tr = self.calc_sec(r_train)

        if self.index_optimize_noise:
            self.index_optimize_noise = self.index_optimize_noise
            self.add_eps_to_sigma = self._add_jiggle_noise
            self.split_hyp_and_noise = self._split_hyp_and_noise
        else:
            self.add_eps_to_sigma = self._add_jiggle
            self.split_hyp_and_noise = lambda theta: [theta, None]

    def _split_hyp_and_noise(self, theta):
        θ = theta[:-1]
        noise = theta[-1]
        return θ, noise

    def trainingK_all(self):
        raise NotImplementedError

    def mixedK_all(self):
        raise NotImplementedError

    def testK_all(self):
        raise NotImplementedError

    def setup_kernel_include_difference_prime(self, K_func):
        """
        Function that construct a kernel that calculates shifted difference of variable at first argument.
        S_{L\boldsymbol{e}^{alpha}}(kernel)
        """

        def K_difprime(r, rp):
            return K_func(r, rp + self.lbox) - K_func(r, rp)

        return K_difprime

    def setup_kernel_include_difference(self, K_func):
        """
        Function that construct a kernel that calculates shifted difference of variable at second argument.
        S^{prime}_{L\boldsymbol{e}^{alpha}}(kernel)
        """

        def K_dif(r, rp):
            return K_func(r + self.lbox, rp) - K_func(r, rp)

        return K_dif

    def setup_kernel_difdif(self, K_func):
        """
        Function that construct a kernel that calculates shifted difference of variable both at first and second argument.
        S_{L\boldsymbol{e}^{alpha}}S^{prime}_{L\boldsymbol{e}^{alpha}}(kernel)
        """

        def K_difdif(r, rp):
            return (
                K_func(r + self.lbox, rp + self.lbox)
                - K_func(r + self.lbox, rp)
                - K_func(r, rp + self.lbox)
                + K_func(r, rp)
            )

        return K_difdif
