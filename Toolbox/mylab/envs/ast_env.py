from rllab.envs.base import Env
from rllab.envs.base import Step
import numpy as np
from mylab.simulators.example_av_simulator import ExampleAVSimulator
from mylab.rewards.example_av_reward import ExampleAVReward
import pdb


class ASTEnv(Env):
    def __init__(self,
                 action_only=True,
                 sample_init_state=False,
                 s_0=None,
                 simulator=None,
                 reward_function=None,
                 spaces=None):
        # Constant hyper-params -- set by user
        self.action_only = action_only
        self.spaces = spaces
        # These are set by reset, not the user
        self._done = False
        self._reward = 0.0
        self._info = []
        self._step = 0
        self._action = None
        self._actions = []
        self._first_step = True

        if s_0 is None:
            self._init_state = self.observation_space.sample()
        else:
            self._init_state = s_0
        self._sample_init_state = sample_init_state
        self.simulator = simulator
        if self.simulator is None:
            self.simulator = ExampleAVSimulator()
        self.reward_function = reward_function
        if self.reward_function is None:
            self.reward_function = ExampleAVReward()

        super().__init__()

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
        if obs is None:
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
        if self._sample_init_state:
            self._init_state = self.observation_space.sample()
        self.simulator.simulate(actions)

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._actions = []
        if self._sample_init_state:
            self._init_state = self.observation_space.sample()

        return self.simulator.reset(self._init_state)

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        return self.spaces.action_space

    @property
    def observation_space(self):
        """
        Returns a Space object
        """

        return self.spaces.observation_space

    def get_cache_list(self):
        return self._info

    def log(self):
        self.simulator.log()


