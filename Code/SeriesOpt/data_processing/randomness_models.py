from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm, uniform
from ..config import Config

class RandomnessModel(ABC):

    @abstractmethod
    def pdf(self, z):
        """
        Probability density function (PDF) of the randomness model at point z.
        """
        pass

    @abstractmethod
    def sample(self):
        """
        Generate a random sample from the distribution (if needed).
        """
        pass

class NormalRandomness(RandomnessModel):
    def __init__(self, sigma):
        self.sigma = sigma
        self.distribution = norm(scale=self.sigma)

    def pdf(self, z):
        return self.distribution.pdf(z)

    def sample(self, nsamples=1):
        return self.distribution.rvs(size=nsamples)
    
    def get_meta_data(self):
        return {'type': 'NormalRandomness', 'sigma': self.sigma}
    
class UniformRandomness(RandomnessModel):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.distribution = uniform(loc=self.a, scale=self.b - self.a)

    def pdf(self, z):
        return self.distribution.pdf(z)

    def sample(self, nsamples=1):
        return self.distribution.rvs(size=nsamples)
    
    def get_meta_data(self):
        return {'type': 'UniformRandomness', 'a': self.a, 'b': self.b}
    
class DiscreteRandomness(RandomnessModel):
    def __init__(self, values, probabilities):
        self.values = values
        self.probabilities = probabilities
        self.cumulative_probabilities = np.cumsum(probabilities)

    def pdf(self, z):
        if z in self.values:
            index = self.values.index(z)
            return self.probabilities[index]
        else:
            return 0

    def sample(self, nsamples=1):
        random_numbers = np.random.rand(nsamples)
        indices = np.searchsorted(self.cumulative_probabilities, random_numbers)
        return [self.values[i] for i in indices]
    
    def get_meta_data(self):
        return {'type': 'DiscreteRandomness', 'values': self.values, 'probabilities': self.probabilities}
    
from scipy.stats import gaussian_kde

class EmpiricalKDERandomness(RandomnessModel):
    """
    Uses a Gaussian KDE to estimate the PDF from empirical residuals.
    """

    def __init__(self, residuals, bw_method='scott'):
        self.residuals = np.asarray(residuals, dtype=float)
        self.kde = gaussian_kde(self.residuals, bw_method=bw_method)
        self.sigma = np.std(self.residuals)

    def pdf(self, z):
        # If z is scalar, we can just do:
        return self.kde.evaluate(z)[0]  # returns an array
        # If z is array, we might do .evaluate(z), returning array

    def sample(self, nsamples=1):
        samples = self.kde.resample(nsamples).flatten()
        return samples

    def get_meta_data(self):
        return {
            'type': 'EmpiricalKDERandomness',
            'bandwidth': self.kde.factor,
            'npoints': len(self.residuals),
        }