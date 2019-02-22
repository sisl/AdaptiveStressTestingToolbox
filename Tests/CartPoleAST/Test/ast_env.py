from cached_property import cached_property
from garage.misc.overrides import overrides
from garage.envs.base import GarageEnv
from garage.envs.env_spec import EnvSpec
from garage.envs.base import Step
from garage.tf.spaces import Box
from garage.tf.spaces import Dict
from garage.tf.spaces import Discrete
from garage.tf.spaces import Tuple

from gym.spaces import Box as GymBox
from gym.spaces import Dict as GymDict
from gym.spaces import Discrete as GymDiscrete
from gym.spaces import Tuple as GymTuple

import numpy as np
from garage.envs.env_spec import EnvSpec
import pdb
import gym
from garage.core import Serializable


class ASTEnv(gym.Env, Serializable):
# class ASTEnv(GarageEnv):
    def __init__(self,
                 open_loop=True,
                 action_only=True,
                 fixed_init_state=False,
                 s_0=None,
                 simulator=None,
                 reward_function=None,
                 spaces=None):
        # Constant hyper-params -- set by user
        self.open_loop=open_loop
        self.action_only = action_only #is this redundant?
        self.spaces = spaces
        # These are set by reset, not the user
        self._done = False
        self._reward = 0.0
        self._info = []
        self._step = 0
        self._action = None
        self._actions = []
        self._first_step = True
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = None

        if s_0 is None:
            self._init_state = self.observation_space.sample()
        else:
            self._init_state = s_0
        self._fixed_init_state = fixed_init_state
        self.simulator = simulator
        self.reward_function = reward_function

        if hasattr(self.simulator, "vec_env_executor") and callable(getattr(self.simulator, "vec_env_executor")):
            self.vectorized = True
        else:
            self.vectorized = False
        # super().__init__(self)
        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        self._action = action
        self._actions.append(action)
        # Update simulation step
        obs = self.simulator.step(self._action)
        if (obs is None) or (self.open_loop is True):
            obs = self._init_state
        if self.simulator.is_goal():
            self._done = True
        # Calculate the reward for this step
        self._reward = self.reward_function.give_reward(
            action=self._action,
            info=self.simulator.get_reward_info())
        # Update instance attributes
        # self.log()
        # if self._step == self.c_max_path_length - 1:
        #     # pdb.set_trace()
        #     self.simulator.simulate(self._actions)
        self._step = self._step + 1

        return Step(observation=obs,
                    reward=self._reward,
                    done=self._done,
                    info={'cache': self._info})

    def simulate(self, actions):
        if not self._fixed_init_state:
            self._init_state = self.observation_space.sample()
        self.simulator.simulate(actions, self._init_state)

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._actions = []
        if not self._fixed_init_state:
            self._init_state = self.observation_space.sample()
        self._done = False
        self._reward = 0.0
        self._info = []
        self._action = None
        self._actions = []
        self._first_step = True

        return self.simulator.reset(self._init_state)

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        if self.spaces is None:
            # return self._to_garage_space(self.simulator.action_space)
            return self.simulator.action_space
        else:
            return self.spaces.action_space

    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        if self.spaces is None:
            # return self._to_garage_space(self.simulator.observation_space)
            return self.simulator.observation_space
        else:
            return self.spaces.observation_space

    def get_cache_list(self):
        return self._info

    def log(self):
        self.simulator.log()

    def render(self):
        if hasattr(self.simulator, "render") and callable(getattr(self.simulator, "render")):
            return self.simulator.render()
        else:
            return None

    def close(self):
        if hasattr(self.simulator, "close") and callable(getattr(self.simulator, "close")):
            self.simulator.close()
        else:
            return None

    def vec_env_executor(self, n_envs, max_path_length):
        return self.simulator.vec_env_executor(n_envs, max_path_length, self.reward_function,
                                               self._fixed_init_state, self._init_state,
                                               self.open_loop)

    def log_diagnostics(self, paths):
        pass

    @cached_property
    @overrides
    def spec(self):
        """
        Returns an EnvSpec.

        Returns:
            spec (garage.envs.EnvSpec)
        """
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space)

    # @overrides
    # def _to_garage_space(self, space):
    #     """
    #     Converts a gym.space to a garage.tf.space.

    #     Returns:
    #         space (garage.tf.spaces)
    #     """
    #     if isinstance(space, GymBox):
    #         return Box(low=space.low, high=space.high)
    #     elif isinstance(space, GymDict):
    #         return Dict(space.spaces)
    #     elif isinstance(space, GymDiscrete):
    #         return Discrete(space.n)
    #     elif isinstance(space, GymTuple):
    #         return Tuple(list(map(self._to_garage_space, space.spaces)))
    #     else:
    #         return space