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
