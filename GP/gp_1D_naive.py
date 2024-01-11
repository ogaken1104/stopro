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

    def trainingKs(self):
        Ks = self.setup_Ks()
        return Ks

    def mixedKs(self):
        Ks = self.setup_Ks()
        return Ks

    def testKs(self):
        Ks = self.setup_Ks()
        return Ks

    def setup_Ks(self):
        def Kyy(r, rp, theta):
            return self.K(r, rp, theta)

        Ks = [[Kyy]]
        return Ks
