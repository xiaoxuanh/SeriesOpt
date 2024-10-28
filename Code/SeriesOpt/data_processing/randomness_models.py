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

    def sample(self):
        return self.distribution.rvs()
    
    def get_meta_data(self):
        return {'type': 'NormalRandomness', 'sigma': self.sigma}
    
class UniformRandomness(RandomnessModel):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.distribution = uniform(loc=self.a, scale=self.b - self.a)

    def pdf(self, z):
        return self.distribution.pdf(z)

    def sample(self):
        return self.distribution.rvs()
    
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

    def sample(self):
        random_number = np.random.rand()
        index = np.searchsorted(self.cumulative_probabilities, random_number)
        return self.values[index]
    
    def get_meta_data(self):
        return {'type': 'DiscreteRandomness', 'values': self.values, 'probabilities': self.probabilities}