from abc import ABC, abstractmethod
from scipy.stats import randint
from numpy.random import default_rng

class Distribution(ABC):

    def __init__(self, seed=None):
        self._rng = default_rng(seed)

    # Read distribution from JSON
    def read(json):
        if json['name'] == 'uniform':
            del json['name']
            return UniformDistribution(**json)
        elif json['name'] == 'discrete':
            del json['name']
            return DiscreteDistribution(**json)
        elif json['name'] == 'normal':
            del json['name']
            return NormalDistribution(**json)
        else:
            raise TypeError(json['name'] + ' is not a recognised distribution')

    @abstractmethod
    def next(self):
        pass


class DiscreteDistribution(Distribution):

    def __init__(self, values, probabilities=None, seed=None):
        super().__init__(seed)

        # Check probabilities size
        if probabilities is not None:
            if len(probabilities) != len(values):
                raise ValueError("Values length: {" + len(values) + "} and probabilities"
                                 " length {" + len(probabilities) + "} are not the same")

        self._values = values

    def next(self):

        # TODO: Need to look at random state - I think this will just produce the same
        # number over and over again because it is having the same seed used over and
        # over
        values_index = randint.rvs(0, len(self._values)-1, random_state=self._rng)
        return self._values[values_index]
