from jax import grad
import jax.numpy as jnp

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

    def setup_kernel(self, theta):
        """Construct Kernels"""

        def Kyy(r, rp):
            return self.K(r, rp, theta)

        def Kyly(r, rp):
            return self.L1K(r, rp, theta)

        def Klyly(r, rp):
            return self.LLK(r, rp, theta)

        return Kyy, Kyly, Klyly

    def trainingK_all(self, theta, train_pts):
        """
        Args :
            θ (jnp.array) : kernel hyperparameters
            training points (list(jnp.array)):
        """
        Kyy, Kyly, Klyly = self.setup_kernel(theta)

        Ks = [[Kyy, Kyly], [Klyly]]

        return self.calculate_K_symmetric(train_pts, Ks)

    def mixedK_all(self, theta, test_pts, train_pts):
        """
        Args :
            θ (jnp.array) : kernel hyperparameters
            test points (list(jnp.array)):
            training points (list(jnp.array)):
        """
        Kyy, Kyly, _ = self.setup_kernel(theta)

        Ks = [
            [Kyy, Kyly],
        ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)

    def testK_all(self, theta, test_pts):
        """
        Args :
            θ (jnp.array) : kernel hyperparameters
            test points (List(jnp.array)): r_test
        """
        Kyy, _, _ = self.setup_kernel(theta)
        Ks = [
            [Kyy],
        ]

        return self.calculate_K_symmetric(test_pts, Ks)
