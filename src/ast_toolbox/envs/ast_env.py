""" Gym environment to turn general AST tasks into garage compatible problems."""
import gym
import numpy as np
from cached_property import cached_property
from garage.envs.base import Step
from garage.envs.env_spec import EnvSpec

from ast_toolbox.rewards import ExampleAVReward
from ast_toolbox.simulators import ExampleAVSimulator


class ASTEnv(gym.Env):
    r""" Gym environment to turn general AST tasks into garage compatible problems.

    Parameters
    ----------
    open_loop : bool
        True if the simulation is open-loop, meaning that AST must generate all actions ahead of time, instead
        of being able to output an action in sync with the simulator, getting an observation back before
        the next action is generated. False to get interactive control, which requires that `blackbox_sim_state`
        is also False.
    blackbox_sim_state : bool
        True if the true simulation state can not be observed, in which case actions and the initial conditions are
        used as the observation. False if the simulation state can be observed, in which case it will be used.
    fixed_init_state : bool
        True if the initial state is fixed, False to sample the initial state for each rollout from the observaation
        space.
    s_0 : array_like
        The initial state for the simulation (ignored if `fixed_init_state` is False)
    simulator : :py:class:`ast_toolbox.simulators.ASTSimulator`
        The simulator wrapper, inheriting from `ast_toolbox.simulators.ASTSimulator`.
    reward_function : :py:class:`ast_toolbox.rewards.ASTReward`
        The reward function, inheriting from `ast_toolbox.rewards.ASTReward`.
    spaces : :py:class:`ast_toolbox.spaces.ASTSpaces`
        The observation and action space definitions, inheriting from `ast_toolbox.spaces.ASTSpaces`.
    """

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
        r"""
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.

        Parameters
        ----------
        action : array_like
            An action provided by the environment.

        Returns
        -------
        : :py:func:`garage.envs.base.Step`
            A step in the rollout.
            Contains the following information:
                - observation (array_like): Agent's observation of the current environment.
                - reward (float): Amount of reward due to the previous action.
                - done (bool): Is the current step a terminal or goal state, ending the rollout.
                - actions (array_like): The action taken at the current.
                - state (array_like): The cloned simulation state at the current cell, used for resetting if chosen to start a rollout.
                - is_terminal (bool): Whether or not the current cell is a terminal state.
                - is_goal (bool): Whether or not the current cell is a goal state.
        """
        self._env_state_before_action = self._env_state.copy()

        self._action = action
        self._actions.append(action)
        action_return = self._action

        # Update simulation step
        obs = self.simulator.step(self._action)
        if (obs is None) or (self.open_loop is True) or (self.blackbox_sim_state):
            obs = np.array(self._init_state)

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

        return Step(observation=obs,
                    reward=self._reward,
                    done=self._done,
                    actions=action_return,
                    state=self._env_state_before_action,
                    is_terminal=self.simulator.is_terminal(),
                    is_goal=self.simulator.is_goal())

    def simulate(self, actions):
        r"""Run a full simulation rollout.

        Parameters
        ----------
        actions : list[array_like]
            A list of array_likes, where each member is the action taken at that step.

        Returns
        -------
        int
            The step of the trajectory where a collision was found, or -1 if a collision was not found.
        dict
            A dictionary of simulation information for logging and diagnostics.
        """
        if not self._fixed_init_state:
            self._init_state = self.observation_space.sample()
        return self.simulator.simulate(actions, self._init_state)

    def reset(self):
        r"""Resets the state of the environment, returning an initial observation.

        Returns
        -------
        observation : array_like
            The initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._actions = []
        if not self._fixed_init_state:
            self._init_state = self.observation_space.sample()
        self._done = False
        self._reward = 0.0
        self._cum_reward = 0.0
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
        r"""Convenient access to the environment's action space.

        Returns
        -------
        : `gym.spaces.Space <https://gym.openai.com/docs/#spaces>`_
            The action space of the reinforcement learning problem.
        """
        if self.spaces is None:
            # return self._to_garage_space(self.simulator.action_space)
            return self.simulator.action_space
        else:
            return self.spaces.action_space

    @property
    def observation_space(self):
        r"""Convenient access to the environment's observation space.

        Returns
        -------
        : `gym.spaces.Space <https://gym.openai.com/docs/#spaces>`_
            The observation space of the reinforcement learning problem.
        """
        if self.spaces is None:
            # return self._to_garage_space(self.simulator.observation_space)
            return self.simulator.observation_space
        else:
            return self.spaces.observation_space

    def log(self):
        r"""Calls the simulator's `log` function.

        """
        self.simulator.log()

    def render(self, **kwargs):
        r"""Calls the simulator's `render` function, if it exists.

        Parameters
        ----------
        kwargs :
            Keyword arguments used in the simulators `render` function.

        Returns
        -------
        None or object
            Returns the output of the simulator's `render` function, or None if the simulator has no `render` function.
        """
        if hasattr(self.simulator, "render") and callable(getattr(self.simulator, "render")):
            return self.simulator.render(**kwargs)
        else:
            return None

    def close(self):
        r"""Calls the simulator's `close` function, if it exists.

        Returns
        -------
        None or object
            Returns the output of the simulator's `close` function, or None if the simulator has no `close` function.
        """
        if hasattr(self.simulator, "close") and callable(getattr(self.simulator, "close")):
            return self.simulator.close()
        else:
            return None

    @cached_property
    def spec(self):
        r"""Returns a garage environment specification.

        Returns
        -------
        :py:class:`garage.envs.env_spec.EnvSpec`
            A garage environment specification.
        """
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space)
