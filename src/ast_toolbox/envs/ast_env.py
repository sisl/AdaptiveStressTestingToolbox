
import gym
import numpy as np
from cached_property import cached_property
from garage.envs.base import Step
from garage.envs.env_spec import EnvSpec

from ast_toolbox.rewards import ExampleAVReward
from ast_toolbox.simulators import ExampleAVSimulator


class ASTEnv(gym.Env):
    # class ASTEnv(GarageEnv):
    def __init__(self,
                 open_loop=True,
                 blackbox_sim_state=True,
                 fixed_init_state=False,
                 s_0=None,
                 simulator=None,
                 reward_function=None,
                 spaces=None):
        # Constant hyper-params -- set by user
        self.open_loop = open_loop
        self.blackbox_sim_state = blackbox_sim_state  # is this redundant?
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
        self._cum_reward = 0.0

        if s_0 is None:
            self._init_state = self.observation_space.sample()
        else:
            self._init_state = s_0
        self._fixed_init_state = fixed_init_state
        self.simulator = simulator
        if self.simulator is None:
            self.simulator = ExampleAVSimulator()
        self.reward_function = reward_function
        if self.reward_function is None:
            self.reward_function = ExampleAVReward()

        if hasattr(self.simulator, "vec_env_executor") and callable(getattr(self.simulator, "vec_env_executor")):
            self.vectorized = True
        else:
            self.vectorized = False
        # super().__init__(self)

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
        self._env_state_before_action = self._env_state.copy()

        self._action = action
        self._actions.append(action)
        action_return = self._action
        # Update simulation step
        obs = self.simulator.step(self._action)
        if (obs is None) or (self.open_loop is True) or (self.blackbox_sim_state):
            obs = np.array(self._init_state)
        # if self.simulator.is_goal():
        if self.simulator.is_terminal() or self.simulator.is_goal():
            self._done = True
        # Calculate the reward for this step
        self._reward = self.reward_function.give_reward(
            action=self._action,
            info=self.simulator.get_reward_info())
        self._cum_reward += self._reward
        # Update instance attributes
        self._step = self._step + 1

        self._simulator_state = self.simulator.clone_state()
        self._env_state = np.concatenate((self._simulator_state,
                                          np.array([self._cum_reward]),
                                          np.array([self._step])),
                                         axis=0)
        # if self._done:
        #     self.simulator.simulate(self._actions, self._init_state)
        #     if not (self.simulator.is_goal() or self.simulator.is_terminal()):
        #         pdb.set_trace()
        return Step(observation=obs,
                    reward=self._reward,
                    done=self._done,
                    sim_info=self._info,
                    actions=action_return,
                    # step = self._step -1,
                    # real_actions=self._action,
                    state=self._env_state_before_action,
                    # root_action=self.root_action,
                    is_terminal=self.simulator.is_terminal(),
                    is_goal=self.simulator.is_goal())

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
        self._cum_reward = 0.0
        self._info = []
        self._action = None
        self._actions = []
        self._first_step = True
        self._step = 0

        obs = np.array(self.simulator.reset(self._init_state))
        if not self._fixed_init_state:
            obs = np.concatenate((obs, np.array(self._init_state)), axis=0)

        self._simulator_state = self.simulator.clone_state()
        self._env_state = np.concatenate((self._simulator_state,
                                          np.array([self._cum_reward]),
                                          np.array([self._step])),
                                         axis=0)

        return obs

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
    def spec(self):
        """
        Returns an EnvSpec.

        Returns:
            spec (garage.envs.EnvSpec)
        """
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space)
