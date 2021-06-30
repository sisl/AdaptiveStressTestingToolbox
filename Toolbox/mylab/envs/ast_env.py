import gym
from garage.core import Serializable
from garage.envs.base import Step
import numpy as np
import pdb

class ASTEnv(gym.Env, Serializable):
    def __init__(self,
                 simulator,
                 reward_function,
                 interactive=True,
                 sample_init_state=False,
                 s_0=None,
                 ):
        # Constant hyper-params -- set by user
        self.interactive = interactive
        # These are set by reset, not the user
        self._done = False
        self._reward = 0.0
        self._info = []
        self._action = None
        self._actions = []
        self._first_step = True

        if s_0 is None:
            self._init_state = self.observation_space.sample()
        else:
            self._init_state = s_0
        self._sample_init_state = sample_init_state

        self.simulator = simulator
        self.reward_function = reward_function

        if hasattr(self.simulator, "vec_env_executor") and callable(getattr(self.simulator, "vec_env_executor")):
            self.vectorized = True
        else:
            self.vectorized = False

        # super().__init__()
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
        if not self.interactive:
            obs = self._init_state
        # if self.simulator.is_goal():
        if self.simulator.isterminal():
            self._done = True
        # Calculate the reward for this step
        self._reward = self.reward_function.give_reward(
            action=self._action,
            info=self.simulator.get_reward_info())

        # self.log()


        return Step(observation=obs,
                    reward=self._reward,
                    done=self._done,
                    info={'cache': self._info})

    def simulate(self, actions):
        if self._sample_init_state:
            self._init_state = self.observation_space.sample()
        return self.simulator.simulate(actions, self._init_state)

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
        self._done = False
        self._reward = 0.0
        self._info = []
        self._action = None
        self._actions = []
        self._first_step = True
        o = self.simulator.reset(self._init_state)

        if self.interactive:
            return o
        else:
            return self._init_state
            
    # def seed(self,seed):
    #     return self.simulator.seed(seed)

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        return self.simulator.action_space

    @property
    def observation_space(self):
        """
        Returns a Space object
        """

        return self.simulator.observation_space

    def get_cache_list(self):
        return self._info

    def log(self):
        self.simulator.log()

    def render(self, mode):
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
        return self.simulator.vec_env_executor(n_envs,max_path_length,self.reward_function,
                                                self._sample_init_state,self._init_state,
                                                self.interactive)

