import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap


class GPmodel:
    """
    The base class for any Gaussian Process model
    """

    def __init__(self, index_optimize_noise: jnp.array = None):
        """
        Args:
            index_optimize_noised (jnp.array): the index of block covariance matrices for optimize the noise
        """
        self.index_optimize_noise = False

    @staticmethod
    def outermap(f):
        return vmap(vmap(f, in_axes=(None, 0, None)), in_axes=(0, None, None))

    def _add_jiggle(
        self,
        Σ: jnp.ndarray,
        jiggle_constant: jnp.ndarray,
        noise_parameter: jnp.ndarray = None,
    ):
        """Add small artificial jiggle to covariance matrix for constant noise.

        noise_parameter is dummy.

        Args:
            Σ (jnp.array): covariance matrix
            jiggle_constant (float): the constant for adding jiggle noise
            noise_parameter (float): dummy

        Returns:
            jnp.array: covariance matrix with jiggle noise
        """
        jiggle = jnp.ones(len(Σ)) * jiggle_constant
        return Σ + jnp.diag(jiggle)

    def _add_jiggle_noise(
        self,
        Σ: jnp.ndarray,
        jiggle_constant: jnp.ndarray,
        noise_parameter: jnp.ndarray = None,
    ):
        """Add small artificial jiggle and noise to covariance matrix for the case we optimize the noise parameter.

        Args:
            Σ (jnp.array): covariance matrix
            jiggle_constant (float): the constant for adding jiggle noise
            noise_parameter (float): the noise parameter for adding noise

        Returns:
            jnp.array: covariance matrix with jiggle noise
        """
        noise_jiggle = jnp.ones(len(Σ))
        r_index_noise_start = self.index_optimize_noise[0]
        r_index_noise_end = self.index_optimize_noise[-1]
        index_noise_start = self.sec_tr[r_index_noise_start]
        index_noise_end = self.sec_tr[r_index_noise_end + 1]

        noise_jiggle = noise_jiggle.at[index_noise_start:index_noise_end].multiply(
            jnp.exp(noise_parameter)
        )
        noise_jiggle = noise_jiggle.at[index_noise_end:].multiply(jiggle_constant)
        return Σ + jnp.diag(noise_jiggle)

    def logpGP(self, δf: jnp.ndarray, Σ: jnp.ndarray):
        """Compute minus log-likelihood of observing δf = f - <f>, for GP with covariance matrix Σ

        Args:
            δf (jnp.array): the difference of the function value and the average
            Σ (jnp.array): the covariance matrix

        Returns:
            jnp.array: minus log-likelihood
        """
        n = len(δf)
        L = jnp.linalg.cholesky(Σ)
        v = jnp.linalg.solve(L, δf)
        return (
            0.5 * jnp.dot(v, v)
            + jnp.sum(jnp.log(jnp.diag(L)))
            + 0.5 * n * jnp.log(2.0 * jnp.pi)
        )

    def postGP(self, δfb, Kaa, Kab, Kbb):
        """Compute posterior average and covariance from conditional GP p(fa | xa, xb, fb)
        [fa,fb] ~ 𝒩([μ_fa, μ_fb], [[Kaa, Kab],[Kab^T, Kbb]])])
        fa|fb   ~ 𝒩(μf + Kab Kbb \ (fb - μ_fb) , Kaa - Kab Kbb \ Kab^T)

        Args:
            δfb (jnp.array): the difference of the function value and the average
            Kaa (jnp.array): the covariance matrix of test points
            Kab (jnp.array): the covariance matrix between test and training points
            Kbb (jnp.array): the covariance matrix of training points

        Returns:
            jnp.array: posterior average
            jnp.array: posterior covariance
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
        V = jnp.linalg.solve(L, jnp.transpose(Kab))  # V = [v_1, v_2, ... ]^t
        Kpost = Kaa - jnp.einsum("ji, jk->ik", V, V)  # V^tV
        return μpost, Kpost  # note should add μ(x*) to average

    def calculate_K_training(self, pts, Ks, theta):
        """Compute training covariance matrix

        Args:
            pts (list[jnp.array]): list of coodinates of training points
            Ks (list[list[function]]): list of list of kernel functions
            theta (jnp.array): kernel hyperparameters

        Returns:
            jnp.array: training covariance matrix
        """
        Σ = jnp.zeros((self.sec_tr[self.num_tr], self.sec_tr[self.num_tr]))
        for i in range(self.num_tr):
            for j in range(i, self.num_tr):
                # upper triangular matrix
                Σ = Σ.at[
                    self.sec_tr[i] : self.sec_tr[i + 1],
                    self.sec_tr[j] : self.sec_tr[j + 1],
                ].set(Ks[i][j - i](pts[i], pts[j], theta))
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

    def calculate_K_test(self, pts, Ks, theta):
        """Compute test covariance matrix

        Args:
            pts (list[jnp.array]): list of coodinates of test points
            Ks (list[list[function]]): list of list of kernel functions
            theta (jnp.array): kernel hyperparameters

        Returns:
            jnp.array: test covariance matrix

        """
        Σ = jnp.zeros((self.sec_te[self.num_te], self.sec_te[self.num_te]))
        for i in range(self.num_te):
            for j in range(i, self.num_te):
                # upper triangular matrix
                Σ = Σ.at[
                    self.sec_te[i] : self.sec_te[i + 1],
                    self.sec_te[j] : self.sec_te[j + 1],
                ].set(Ks[i][j - i](pts[i], pts[j], theta))
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

    def calculate_K_asymmetric(self, train_pts, test_pts, Ks, theta):
        """Compute asymmetric part of covariance matrix (mixed_K)

        Args:
            train_pts (list[jnp.array]): list of coodinates of training points
            test_pts (list[jnp.array]): list of coodinates of test points
            Ks (list[list[function]]): list of list of kernel functions
            theta (jnp.array): kernel hyperparameters

        Returns:
            jnp.array: mixed covariance matrix

        """
        Σ = jnp.zeros((self.sec_te[self.num_te], self.sec_tr[self.num_tr]))
        for i in range(self.num_te):
            for j in range(self.num_tr):
                Σ = Σ.at[
                    self.sec_te[i] : self.sec_te[i + 1],
                    self.sec_tr[j] : self.sec_tr[j + 1],
                ].set(Ks[i][j](test_pts[i], train_pts[j], theta))
        return Σ

    def trainingFunction_all(self, theta, *args):
        """Returns minus log-likelihood given Kernel hyperparamters θ and training data args

        Args:
            theta: kernel hyperparameters
            args: training input, delta_y, jiggle parameter
        """
        r, delta_y, ϵ = args
        θ, noise = self.split_hyp_and_noise(theta)
        Σ = self.trainingK_all(θ, r)
        Σ = self.add_eps_to_sigma(Σ, ϵ, noise_parameter=noise)
        return self.logpGP(delta_y, Σ)

    def predictingFunction_all(self, theta, *args):
        """Returns conditional posterior average and covariance matrix given Kernel hyperparamters θ  and test and training data

        Args:
            theta: kernel hyperparameters
            args: test input, test average, training input, delta_y, jiggle parameter

        Returns:
            μpost (jnp.ndarray): [μux,μuy,μp]
            Σpost (jnp.ndarray): [Σux,Σuy,Σp]
        """
        r_test, μ_test, r_train, delta_y_train, ϵ = args
        θ, noise = self.split_hyp_and_noise(theta)
        Σbb = self.trainingK_all(θ, r_train)
        Σab = self.mixedK_all(θ, r_test, r_train)
        Σbb = self.add_eps_to_sigma(Σbb, ϵ, noise_parameter=noise)
        Σaa = self.testK_all(θ, r_test)
        # create single training array, with velocities and forces (second derivatives)
        μposts, Σposts = self.postGP(delta_y_train, Σaa, Σab, Σbb)
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
        """Calculate the section of the covariance matrix for each point"""
        sec = np.concatenate([np.zeros(1, dtype=int), np.cumsum([len(x) for x in pts])])
        return sec

    def set_constants(self, *args, only_training=False):
        """Set constants for the model"""
        if only_training:
            r_train, delta_y_train, ϵ = args
        else:
            r_test, μ_test, r_train, delta_y_train, ϵ = args
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

    def trainingK_all(self, theta, train_pts):
        """
        Args:
            theta (jnp.ndarray) : kernel hyperparameters
            train_pts (list[jnp.ndarray]) : list of coodinates of training points

        Retruns:
            jnp.ndarray: traininc covariance matrix
        """
        Ks = self.trainingKs

        return self.calculate_K_training(train_pts, Ks, theta)

    def mixedK_all(self, theta, test_pts, train_pts):
        Ks = self.mixedKs
        return self.calculate_K_asymmetric(train_pts, test_pts, Ks, theta)

    def testK_all(self, theta, test_pts):
        Ks = self.testKs
        return self.calculate_K_test(test_pts, Ks, theta)

    def setup_trainingKs(self):
        raise NotImplementedError

    def setup_mixedKs(self):
        raise NotImplementedError

    def setup_testKs(self):
        raise NotImplementedError

    def setup_all_Ks(self):
        self.setup_trainingKs()
        self.setup_mixedKs()
        self.setup_testKs()

    def setup_Ks_dKdtheta(self):
        r"""Setup \frac{dK}{dtheta} for training covariance matrix"""

        self.Ks_dKdtheta = []
        for _K_row in self.trainingKs:
            dKdtheta_row = []
            for _K in _K_row:
                dKdtheta_row.append(jax.jacfwd(_K, 2))
            self.Ks_dKdtheta.append(dKdtheta_row)

    def calc_dKdtheta(self, theta, train_pts):
        r"""Compute \frac{dK}{dtheta} for training covariance matrix"""
        dKdtheta = jnp.zeros(
            (
                self.sec_tr[self.num_tr],
                self.sec_tr[self.num_tr],
                len(theta),
            )
        )
        for i in range(self.num_tr):
            for j in range(i, self.num_tr):
                # upper triangular matrix
                dKdtheta = dKdtheta.at[
                    self.sec_tr[i] : self.sec_tr[i + 1],
                    self.sec_tr[j] : self.sec_tr[j + 1],
                ].set(self.Ks_dKdtheta[i][j - i](train_pts[i], train_pts[j], theta))
                if not j == i:
                    # transpose
                    dKdtheta = dKdtheta.at[
                        self.sec_tr[j] : self.sec_tr[j + 1],
                        self.sec_tr[i] : self.sec_tr[i + 1],
                    ].set(
                        jnp.transpose(
                            dKdtheta[
                                self.sec_tr[i] : self.sec_tr[i + 1],
                                self.sec_tr[j] : self.sec_tr[j + 1],
                            ],
                            (1, 0, 2),
                        )
                    )
        return dKdtheta

    def calc_K_given_theta_i(self, theta_i, theta_other, index_theta, r, ϵ):
        theta = jnp.zeros(len(theta_other) + 1)
        theta = theta.at[index_theta].set(theta_i)
        theta = theta.at[:index_theta].set(theta_other[:index_theta])
        theta = theta.at[index_theta + 1 :].set(theta_other[index_theta:])
        θ, noise = self.split_hyp_and_noise(theta)
        Σ = self.trainingK_all(θ, r)
        Σ = self.add_eps_to_sigma(Σ, ϵ, noise_parameter=noise)
        return Σ

    def setup_kernel_include_difference_prime(self, K_func):
        """
        Function that construct a kernel that calculates shifted difference of variable at first argument.
        S_{L\boldsymbol{e}^{alpha}}(kernel)
        """

        def K_difprime(r, rp, theta):
            return K_func(r, rp + self.lbox, theta) - K_func(r, rp, theta)

        return K_difprime

    def setup_kernel_include_difference(self, K_func):
        """
        Function that construct a kernel that calculates shifted difference of variable at second argument.
        S^{prime}_{L\boldsymbol{e}^{alpha}}(kernel)
        """

        def K_dif(r, rp, theta):
            return K_func(r + self.lbox, rp, theta) - K_func(r, rp, theta)

        return K_dif

    def setup_kernel_difdif(self, K_func):
        """
        Function that construct a kernel that calculates shifted difference of variable both at first and second argument.
        S_{L\boldsymbol{e}^{alpha}}S^{prime}_{L\boldsymbol{e}^{alpha}}(kernel)
        """

        def K_difdif(r, rp, theta):
            return (
                K_func(r + self.lbox, rp + self.lbox, theta)
                - K_func(r + self.lbox, rp, theta)
                - K_func(r, rp + self.lbox, theta)
                + K_func(r, rp, theta)
            )

        return K_difdif

    def d_trainingFunction_all(self, theta, *args):
        """Returns minus log-likelihood given Kernel hyperparamters θ and training data args
        args = velocity position, velocity average, velocity values,
            force position, force average, force values,
            jiggle parameter
        """
        # r,μ,f,ϵ=args
        r, δy, ϵ = args
        θ, noise = self.split_hyp_and_noise(theta)

        def calc_trainingK(theta):
            θ, noise = self.split_hyp_and_noise(theta)
            Σ = self.trainingK_all(θ, r)
            Σ = self.add_eps_to_sigma(Σ, ϵ, noise_parameter=noise)
            return Σ

        ## calc matrix solve K^{-1}y
        Σ = calc_trainingK(theta)
        L = jnp.linalg.cholesky(Σ)
        I = jnp.eye(len(δy))
        Σ_inv = jnp.linalg.solve(L.T, jnp.linalg.solve(L, I))
        α = jnp.linalg.solve(L.transpose(), jnp.linalg.solve(L, δy))
        del L, Σ, I
        ########### previous version #################
        # # dKdtheta = jax.jacfwd(calc_trainingK)(theta)
        # dKdtheta = self.calc_dKdtheta(theta, r)
        # dKdtheta = jnp.transpose(dKdtheta, (2, 0, 1))

        # ## calc first term of loss y^TK^{-1}\frac{dK}{d\theta}K^{-1}y
        # first_term = jnp.einsum(
        #     "j, ij -> i", α.T, jnp.einsum("ijk, k ->ij", dKdtheta, α)
        # )

        # ## calc second term of loss Tr(K^{-1}\frac{dK}{d\theta})
        # second_term = jnp.sum(
        #     jnp.diagonal(jnp.einsum("jk, ikl->ijl", Σ_inv, dKdtheta), axis1=1, axis2=2),
        #     axis=1,
        # )
        #########################################
        # print(f"first_term: {first_term[0]:.3e}, second_term: {second_term[0]:.3e}")
        dloss = jnp.zeros(len(theta))
        # dloss = []

        for index_theta in range(len(theta)):
            # v1
            theta_i = theta[index_theta]
            theta_other = jnp.delete(theta, index_theta)
            calc_dKdtheta_i = jax.jacfwd(self.calc_K_given_theta_i, argnums=0)
            dKdtheta_i = calc_dKdtheta_i(theta_i, theta_other, index_theta, r, ϵ)

            # # v2こちらの方が，BBMMの際にメモリを節約できるが，コンパイルに非常に非常に時間がかかる This compilation takes too long time##
            # ## ノイズを最適化する場合，dKdthetaは，最適化するノイズに該当する対角成分が1，それ以外が0の行列となる
            # if self.index_optimize_noise and index_theta == len(theta) - 1:
            #     dKdtheta_i = jnp.zeros(
            #         (self.sec_tr[self.num_tr], self.sec_tr[self.num_tr])
            #     )
            #     r_index_noise_start = self.index_optimize_noise[0]
            #     r_index_noise_end = self.index_optimize_noise[-1]
            #     index_noise_start = self.sec_tr[r_index_noise_start]
            #     index_noise_end = self.sec_tr[r_index_noise_end + 1]
            #     dKdtheta_i = dKdtheta_i.at[index_noise_start:index_noise_end].set(
            #         jnp.exp(1.0)
            #     )
            # else:
            #     dKss_i = self.setup_dKss_theta_i(index_theta)
            #     dKdtheta_i = self.calculate_K_training(r, dKss_i, θ)
            # #########################################

            first_term = jnp.dot(α.T, jnp.matmul(dKdtheta_i, α))
            second_term = jnp.sum(
                jnp.diagonal(
                    jnp.matmul(Σ_inv, dKdtheta_i),
                )
            )
            dloss = dloss.at[index_theta].set((-first_term + second_term) / 2)

        return dloss
        # return (-first_term + second_term) / 2

    def d_logposterior(self, theta, *args):
        loglikelihood = self.d_trainingFunction_all(theta, *args)
        return loglikelihood + 1.0  # gradient of jeffery's prior sum(theta)

    def _calc_yKinvy(self, theta, *args):
        """Returns y^TK^{-1}y given Kernel hyperparamters θ and training data args

        for bbmm test only
        """
        # r,μ,f,ϵ=args
        r, delta_y, ϵ = args
        θ, noise = self.split_hyp_and_noise(theta)
        Σ = self.trainingK_all(θ, r)
        Σ = self.add_eps_to_sigma(Σ, ϵ, noise_parameter=noise)
        n = len(delta_y)
        L = jnp.linalg.cholesky(Σ)
        v = jnp.linalg.solve(L, delta_y)
        return jnp.dot(v, v)

    def _calc_yKdKKy(self, theta, *args):
        """Returns y^TK^{-1}y given Kernel hyperparamters θ and training data args

        for bbmm test only.
        """
        # r,μ,f,ϵ=args
        r, delta_y, ϵ = args
        θ, noise = self.split_hyp_and_noise(theta)

        def calc_trainingK(theta):
            θ, noise = self.split_hyp_and_noise(theta)
            Σ = self.trainingK_all(θ, r)
            Σ = self.add_eps_to_sigma(Σ, ϵ, noise_parameter=noise)
            return Σ

        dKdtheta = jax.jacfwd(calc_trainingK)(θ)
        dKdtheta = jnp.transpose(dKdtheta, (2, 0, 1))

        ## calc matrix solve K^{-1}y
        Σ = calc_trainingK(θ)
        L = jnp.linalg.cholesky(Σ)
        α = jnp.linalg.solve(L.transpose(), jnp.linalg.solve(L, delta_y))

        ## calc first term of loss y^TK^{-1}\frac{dK}{d\theta}K^{-1}y
        first_term = jnp.einsum(
            "j, ij -> i", α.T, jnp.einsum("ijk, k ->ij", dKdtheta, α)
        )
        return first_term

    # def generate_dK_wrt_theta_i(self, _K, index_theta):
    #     def dK_wrt_theta_i(r, rp, theta):
    #         theta_i = theta[index_theta]
    #         theta_other = jnp.delete(theta, index_theta)

    #         def K_given_theta_i(theta_i, theta_other, index_theta, r, rp):
    #             theta = jnp.zeros(len(theta_other) + 1)
    #             theta = theta.at[index_theta].set(theta_i)
    #             theta = theta.at[:index_theta].set(theta_other[:index_theta])
    #             theta = theta.at[index_theta + 1 :].set(theta_other[index_theta:])
    #             return _K(r, rp, theta)

    #         return jax.jacfwd(K_given_theta_i, argnums=0)(
    #             theta_i, theta_other, index_theta, r, rp
    #         )[0]

    #     return dK_wrt_theta_i

    # def setup_dKss_theta_i(
    #     self,
    #     index_theta,
    # ):
    #     dKss = []
    #     for Ks in self.trainingKs:
    #         dKss_row = []
    #         for _K in Ks:
    #             dK_wrt_theta_i = self.generate_dK_wrt_theta_i(_K, index_theta)
    #             dKss_row.append(copy.copy(dK_wrt_theta_i))
    #         dKss.append(dKss_row)
    #     return dKss
