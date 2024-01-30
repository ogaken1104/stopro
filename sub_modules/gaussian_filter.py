import numpy as np


class GaussianFilter:
    def __init__(self, length: float = 0.1):
        self.length = length

    def kernel(self, r, rp):
        if r.ndim == 0:
            return np.exp(-((r - rp) ** 2) / (2 * self.length**2))
        else:
            return np.exp(-np.sum((r - rp) ** 2, axis=1) / (2 * self.length**2))

    def interpolate(self, r_pred, r_basis, f_basis):
        f_pred = []
        for r_pre in r_pred:
            alpha = self.kernel(r_pre, r_basis)
            f_pred.append(np.sum(alpha * f_basis) / np.sum(alpha))
        return f_pred
