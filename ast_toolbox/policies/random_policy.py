import numpy as np
import tensorflow as tf

from garage.misc import logger
from garage.tf.misc import tensor_utils
from garage.tf.policies import Policy
from garage.tf.spaces import Box,Discrete

from gym.utils import seeding

class RandomDistribution:
    def __init__(self, space, seed=None):
        self.space = space
        self.seed(seed)
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def sample(self, batch_size):
        if isinstance(self.space, Box):
            low = self.space.low
            high = self.space.high
            samples = self.np_random.uniform(
                low=low, high=high,
                size=(batch_size,)+low.shape).astype(np.float32)
        elif isinstance(self.space, Discrete):
            n = self.space.n
            samples = self.np_random.randint(self.n,size=batch_size)
        else:
            raise NotImplementedError
        return samples

    @property
    def dist_info_keys(self):
        return []

    def entropy(self, info=None):
        if isinstance(self.space, Box):
            low = self.space.low
            high = self.space.high
            return np.sum(np.log(h-l) for l,h in zip(low,high))
        elif self.space == Discrete:
            return np.log(self.space.n)


class RandomPolicy(Policy):
    def __init__(self,
                 env_spec,
                 seed=None,
                 name="RandomPolicy"):

        super(RandomPolicy, self).__init__(env_spec)
        self.name = name
        self._dist = RandomDistribution(self.action_space)
        self.seed(seed)

    def seed(self, seed=None):
        self.distribution.seed(seed)

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        actions, _ = self.get_actions([observation])
        return actions[0], dict()

    def get_actions(self, observations):
        batch_size = self.observation_space.flatten(observations).shape[0]
        actions = self.distribution.sample(batch_size)
        return actions, dict()

    def log_diagnostics(self, paths):
        pass

    @property
    def distribution(self):
        return self._dist