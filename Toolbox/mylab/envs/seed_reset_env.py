import gym
import gym.spaces
import numpy as np

from garage.core import Serializable
from garage.envs.util import flat_dim, flatten, unflatten
from garage.misc.overrides import overrides


class SeedResetEnv(gym.Wrapper, Serializable):
    def __init__(
            self,
            env,
            random_reset=True,
            reset_seed=None,
    ):
        super().__init__(env)

        self.random_reset = random_reset
        self.reset_seed = reset_seed
        Serializable.quick_init(self, locals())

    @overrides
    def reset(self, **kwargs):
        if self.random_reset:
            return self.env.reset(**kwargs)
        else:
            self.env.seed(self.reset_seed)
            return self.env.reset(**kwargs)

    @overrides
    def step(self, action):
        return self.env.step(action)

    def log_diagnostics(self, paths):
        pass

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    @overrides
    def max_episode_steps(self):
        return self.env.spec.max_episode_steps
