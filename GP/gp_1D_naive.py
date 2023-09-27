from stopro.GP.gp_1D import GPmodel1D


class GPmodel1DNaive(GPmodel1D):
    """
    The base class for 1D gaussian process with naive
    """

    def __init__(
        self,
        Kernel: callable = None,
        index_optimize_noise: list = None,
    ):
        """
        Args:
            Kernel (callable): kernel function to use
        """
        super().__init__()
        self.Kernel = Kernel
        self.K = self.outermap(Kernel)
        self.index_optimize_noise = index_optimize_noise

    # def split_hyperparams(func):
    #     def wrapper(self, *args):
    #         theta = args[1]
    #         func(self, theta, *args[1:])

    #     return wrapper

    def trainingK_all(self, theta, train_pts):
        """
        Args :
            θ (jnp.array) : kernel hyperparameters
            training points (list(jnp.array)):
        """
        Ks = self.setup_Ks(theta)

        return self.calculate_K_training(train_pts, Ks)

    def mixedK_all(self, theta, test_pts, train_pts):
        """
        Args :
            θ (jnp.array) : kernel hyperparameters
            test points (list(jnp.array)):
            training points (list(jnp.array)):
        """
        Ks = self.setup_Ks(theta)

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, theta, test_pts):
        """
        Args :
            θ (jnp.array) : kernel hyperparameters
            test points (List(jnp.array)): r_test
        """
        Ks = self.setup_Ks(theta)

        return self.calculate_K_test(test_pts, Ks)

    def setup_Ks(self, theta):
        def Kyy(r, rp):
            return self.K(r, rp, theta)

        Ks = [[Kyy]]
        return Ks
