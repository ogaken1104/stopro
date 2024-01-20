from stopro.GP.gp_1D import GPmodel1D


class GPmodelNaive(GPmodel1D):
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
        self.setup_all_Ks()

    # def split_hyperparams(func):
    #     def wrapper(self, *args):
    #         theta = args[1]
    #         func(self, theta, *args[1:])

    #     return wrapper

    def setup_trainingKs(self):
        self.trainingKs = self.setup_Ks()

    def setup_mixedKs(self):
        self.mixedKs = self.setup_Ks()

    def setup_testKs(self):
        self.testKs = self.setup_Ks()

    def setup_Ks(self):
        def Kyy(r, rp, theta):
            return self.K(r, rp, theta)

        Ks = [[Kyy]]
        return Ks
