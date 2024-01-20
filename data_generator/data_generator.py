import numpy as np


class DataGenerator:
    def __init__(self):
        self.r = None
        self.f = None

    def generate_training_data(self):
        """Generate training data"""
        pass

    def load_train(self):
        """Load train data"""
        return self.r, self.f

    def plot_train(self):
        """Plot training data"""
        pass

    def add_white_noise(self, f, sigma2_noise):
        if self.seed is not None:
            np.random.seed(self.seed)
        noise = np.random.normal(0, np.sqrt(sigma2_noise), f.shape)
        noisy_f = f + noise
        return noisy_f

    def delete_out_domain(self, r, radius_min=None):
        index_in_domain = self.get_index_in_domain(r, radius_min)
        return r[index_in_domain]

    def change_outside_values_to_zero(self, r, f):
        is_inside_the_domain = self.get_index_in_domain(r)
        f_new = np.zeros(len(f))
        f_new[is_inside_the_domain] = f[is_inside_the_domain]
        return f_new

    def get_index_in_domain(self, r):
        raise NotImplementedError
