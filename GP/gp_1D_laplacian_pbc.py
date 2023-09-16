import numpy as np
from stopro.GP.gp_1D_laplacian import GPmodel1DLaplacian


class GPmodel1DLaplacianPbc(GPmodel1DLaplacian):
    """
    GP using pbc for 1D gaussian process with laplacian
    """

    def __init__(self, Kernel: callable = None, lbox: float = np.pi * 2):
        super().__init__(Kernel)
        self.lbox = lbox

    def trainingK_all(self, theta, train_pts):
        """
        Args :
            θ (jnp.array) : kernel hyperparameters
            training points (list(jnp.array)):
        """
        Kyy, Kyly, Klyly = self.setup_kernel(theta)
        Kypbcy = self.setup_kernel_include_difference_prime(Kyy)
        Kpbcypbcy = self.setup_kernel_difdif(Kyy)
        Kpbcyly = self.setup_kernel_include_difference(Kyly)

        Ks = [[Kyy, Kypbcy, Kyly], [Kpbcypbcy, Kpbcyly], [Klyly]]

        return self.calculate_K_symmetric(train_pts, Ks)

    def mixedK_all(self, theta, test_pts, train_pts):
        """
        Args :
            θ (jnp.array) : kernel hyperparameters
            test points (list(jnp.array)):
            training points (list(jnp.array)):
        """
        Kyy, Kyly, _ = self.setup_kernel(theta)
        Kypbcy = self.setup_kernel_include_difference_prime(Kyy)

        Ks = [
            [Kyy, Kypbcy, Kyly],
        ]

        return self.calculate_K_asymmetric(train_pts, test_pts, Ks)
