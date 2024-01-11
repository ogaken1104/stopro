import jax.numpy as jnp
from jax import grad

from stopro.GP.gp_1D import GPmodel1D


class GPmodel1DLaplacian(GPmodel1D):
    """
    The base class for 1D gaussian process with laplacian
    """

    def __init__(
        self,
        Kernel: callable = None,
    ):
        """
        Args:
            Kernel (callable): kernel function to use
        """
        super().__init__()
        self.Kernel = Kernel
        self.K = self.outermap(Kernel)
        self.setup_differential_operators()

    def Kyy(self, r, rp, theta):
        return self.K(r, rp, theta)

    def Kyly(self, r, rp, theta):
        return self.L1K(r, rp, theta)

    def Klyly(self, r, rp, theta):
        return self.LLK(r, rp, theta)

    def setup_differential_operators(self):
        # define operators
        def L(f, i):
            return grad(grad(f, i), i)  # L = d^2 / dx^2

        _L0K = L(self.Kernel, 0)  # (L_2 K)(x*,x')
        _L1K = L(self.Kernel, 1)  # (L_2 K)(x*,x')
        _LLK = L(_L0K, 1)

        self.L0K = self.outermap(_L0K)
        self.L1K = self.outermap(_L1K)
        self.LLK = self.outermap(_LLK)

    def trainingKs(self):
        Ks = [[self.Kyy, self.Kyly], [self.Klyly]]
        return Ks

    def mixedKs(self):
        Ks = [
            [self.Kyy, self.Kyly],
        ]
        return Ks

    def testKs(self):
        Ks = [
            [self.Kyy],
        ]
        return Ks
