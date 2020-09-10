"""Gym environment to turn general AST tasks into garage compatible problems with Go-Explore style resets."""
import pdb
import pickle
import random
import shelve

import gym
import numpy as np
from bsddb3 import db
from cached_property import cached_property
from garage.envs.base import Step
from garage.envs.env_spec import EnvSpec

from ast_toolbox.rewards import ExampleAVReward
from ast_toolbox.simulators import ExampleAVSimulator
from ast_toolbox.spaces import ExampleAVSpaces


class Parameterized:
    r"""A slimmed down version of the (deprecated) Parameterized class from garage for passing parameters to
    environments.

    Garage uses pickle to handle parallelization, which limits the types of objects that can be used as class
    attributes withing the environment. This class is a workaround, so that the parallel environments can have
    access to things like a database.

    """

    def __init__(self):
        self._cached_params = {}
        self._cached_param_dtypes = {}
        self._cached_param_shapes = {}
        self._cached_assign_ops = {}
        self._cached_assign_placeholders = {}

    def get_params_internal(self, **tags):
        r"""Internal method to be implemented which does not perform caching

        Parameters
        ----------
        tags : str
            Names of the paramters to return.
        """
        raise NotImplementedError

    def get_params(self, **tags):
        r"""Get the list of parameters, filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'

        Parameters
        ----------
        tags : str
            Names of the paramters to return.
        """
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(**tags)
        return self._cached_params[tag_tuple]


class GoExploreParameter():
    """A wrapper for variables that will be set as parameters in the `GoExploreASTEnv`
    Parameters
    ----------
    name : str
        Name of the parameter.
    value : value
        Value of the parameter.
    """

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def get_value(self, **kwargs):
        r"""Return the value of the parameter.

        Parameters
        ----------
        kwargs :
            Extra keyword arguments (Not currently used).

        Returns
        -------
        object
            The value of the parameter.
        """
        return self.value

    def set_value(self, value):
        r"""Set the value of the parameter

        Parameters
        ----------
        value : object
            What to set the parameters `value` to.
        """
        self.value = value


class GoExploreASTEnv(gym.Env, Parameterized):
    r"""Gym environment to turn general AST tasks into garage compatible problems with Go-Explore style resets.

    Certain algorithms, such as Go-Explore and the Backwards Algorithm, require deterministic resets of the
    simulator. `GoExploreASTEnv` handles this by cloning simulator states and saving them in a cell structure. The
    cells are then stored in a hashed database.

    Parameters
    ----------
    open_loop : bool
        True if the simulation is open-loop, meaning that AST must generate all actions ahead of time, instead
        of being able to output an action in sync with the simulator, getting an observation back before
        the next action is generated. False to get interactive control, which requires that `blackbox_sim_state`
        is also False.
    blackbox_sim_state : bool
        True if the true simulation state can not be observed, in which case actions and the initial conditions are
        used as the observation. False if the simulation state can be observed, in which case it will be used
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

        # gym_env = gym.make('ast_toolbox:GoExploreAST-v0', {'test':'test string'})
        # pdb.set_trace()
        # super().__init__(gym_env)
        # Constant hyper-params -- set by user
        self.open_loop = open_loop
        self.blackbox_sim_state = blackbox_sim_state  # is this redundant?
        self.spaces = spaces
        if spaces is None:
            self.spaces = ExampleAVSpaces()
        # These are set by reset, not the user
        self._done = False
        self._reward = 0.0
        self._info = {}
        self._step = 0
        self._action = None
        self._actions = []
        self._first_step = True
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = None
        self.spec._entry_point = []
        self._cum_reward = 0.0
        self.root_action = None
        self.sample_limit = 10000

        self.simulator = simulator
        if self.simulator is None:
            self.simulator = ExampleAVSimulator()

        if s_0 is None:
            self._init_state = self.observation_space.sample()
        else:
            self._init_state = s_0
        self._fixed_init_state = fixed_init_state

        self.reward_function = reward_function
        if self.reward_function is None:
            self.reward_function = ExampleAVReward()

        if hasattr(self.simulator, "vec_env_executor") and callable(getattr(self.simulator, "vec_env_executor")):
            self.vectorized = True
        else:
            self.vectorized = False
        # super().__init__(self)
        # Always call Serializable constructor last
        self.params_set = False
        self.db_filename = 'database.dat'
        self.key_list = []
        self.max_value = 0
        self.robustify_state = []
        self.robustify = False

        Parameterized.__init__(self)

    def sample(self, population):
        r"""Sample a cell from the cell pool with likelihood proportional to cell fitness.

        The sampling is done using Stochastic Acceptance [1]_, with inspiration from John B Nelson's blog [2]_.

        The sampler rejects cells until the acceptance criterea is met. If the maximum number of rejections is
        exceeded, the sampler then will sample uniformly sample a cell until it finds a cell with fitness > 0. If
        the second sampling phase also exceeds the rejection limit, then the function raises an exception.

        Parameters
        ----------
        population : list
            A list containing the population of cells to sample from.

        Returns
        -------
        object
            The sampled cell.

        Raises
        ------
        ValueError
            If the maximum number of rejections is exceeded in both the proportional and the uniform sampling phases.

        References
        ----------
        .. [1] Lipowski, Adam, and Dorota Lipowska. "Roulette-wheel selection via stochastic acceptance."
        Physica A: Statistical Mechanics and its Applications 391.6 (2012): 2193-2196.
        `<https://arxiv.org/pdf/1109.3627.pdf>`_
        .. [2] `<https://jbn.github.io/fast_proportional_selection/>`_
        """
        attempts = 0
        while attempts < self.sample_limit:
            attempts += 1
            candidate = population[random.choice(self.p_key_list.value)]
            if random.random() < (candidate.fitness / self.p_max_value.value):
                return candidate
        attempts = 0
        while attempts < self.sample_limit:
            attempts += 1
            candidate = population[random.choice(self.p_key_list.value)]
            if candidate.fitness > 0:
                print("Returning Uniform Random Sample - Max Attempts Reached!")
                return candidate
        print("Failed to find a valid state for reset!")
        raise ValueError
        # return population[random.choice(self.p_key_list.value)]

    def get_first_cell(self):
        r"""Returns a the observation and state of the initial state, to be used for a root cell.

        Returns
        -------
        obs : array_like
            Agent's observation of the current environment.
        state : array_like
            The cloned simulation state at the current cell, used for resetting if chosen to start a rollout.
        """

        obs = self.env_reset()
        if self.blackbox_sim_state:
            obs = self.simulator.get_first_action()

        state = np.concatenate((self.simulator.clone_state(),
                                np.array([self._cum_reward]),
                                np.array([-1])),
                               axis=0)

        return obs, state

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
                - cache (dict): A dictionary containing other diagnostic information from the current step.
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

        # Add step number to differentiate identical actions
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
                    cache=self._info,
                    actions=action_return,
                    state=self._env_state_before_action,
                    root_action=self.root_action,
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

    def reset(self, **kwargs):
        r"""Resets the state of the environment, returning an initial observation.

        The reset has 2 modes.

        In the "robustify" mode (self.p_robustify_state.value is not None), the simulator resets
        the environment to `p_robustify_state.value`. It then returns the initial condition.

        In the "Go-Explore" mode, the environment attempts to sample a cell from the cell pool. If successful,
        the simulator is reset to the cell's state. On an error, the environment is reset to the intial state.

        Returns
        -------
        observation : array_like
            The initial observation of the space. (Initial reward is assumed to be 0.)
        """

        try:
            # print(self.p_robustify_state.value)
            if self.p_robustify_state is not None and self.p_robustify_state.value is not None and len(
                    self.p_robustify_state.value) > 0:
                state = self.p_robustify_state.value
                # print('-----------Robustify Init-----------------')
                # print('-----------Robustify Init: ', state, ' -----------------')
                self.simulator.restore_state(state[:-2])
                obs = self.simulator.observation_return()
                self._done = False
                self._cum_reward = state[-2]
                self._step = state[-1]
                # pdb.set_trace()

                self.robustify = True

                self._simulator_state = self.simulator.clone_state()
                self._env_state = np.concatenate((self._simulator_state,
                                                  np.array([self._cum_reward]),
                                                  np.array([self._step])),
                                                 axis=0)
                return self._init_state

            flag = db.DB_RDONLY
            pool_DB = db.DB()
            pool_DB.open(self.p_db_filename.value, dbname=None, dbtype=db.DB_HASH, flags=flag)
            dd_pool = shelve.Shelf(pool_DB, protocol=pickle.HIGHEST_PROTOCOL)
            cell = self.sample(dd_pool)
            dd_pool.close()
            pool_DB.close()

            if cell.state is not None:
                # pdb.set_trace()
                if np.all(cell.state == 0):
                    print("-------DEFORMED CELL STATE-------")
                    obs = self.env_reset()
                else:
                    self.simulator.restore_state(cell.state[:-2])
                    if self.simulator.is_terminal() or self.simulator.is_goal():
                        print('-------SAMPLED TERMINAL STATE-------')
                        pdb.set_trace()
                        obs = self.env_reset()

                    else:
                        if cell.score == 0.0 and cell.parent is not None:
                            print("Reset to cell with score 0.0 ---- terminal: ", self.simulator.is_terminal(),
                                  " goal: ", self.simulator.is_goal(), " obs: ", cell.observation)
                        obs = self.simulator.observation_return()
                        self._done = False
                        self._cum_reward = cell.state[-2]
                        self._step = cell.state[-1]
                        self.root_action = cell.observation
            else:
                print("Reset from start")
                obs = self.env_reset()

            self._simulator_state = self.simulator.clone_state()
            self._env_state = np.concatenate((self._simulator_state,
                                              np.array([self._cum_reward]),
                                              np.array([self._step])),
                                             axis=0)
            # pdb.set_trace()
        except db.DBBusyError:
            print("DBBusyError")
            obs = self.env_reset()
        except db.DBLockNotGrantedError or db.DBLockDeadlockError:
            print("db.DBLockNotGrantedError or db.DBLockDeadlockError")
            obs = self.env_reset()
        except db.DBForeignConflictError:
            print("DBForeignConflictError")
            obs = self.env_reset()
        except db.DBAccessError:
            print("DBAccessError")
            obs = self.env_reset()
        except db.DBPermissionsError:
            print("DBPermissionsError")
            obs = self.env_reset()
        except db.DBNoSuchFileError:
            print("DBNoSuchFileError")
            obs = self.env_reset()
        except db.DBError:
            print("DBError")
            obs = self.env_reset()
        except BaseException:
            print("Failed to get state from database")
            pdb.set_trace()
            obs = self.env_reset()

        return obs

    def env_reset(self):
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
        self._info = {'actions': []}
        self._action = self.simulator.get_first_action()
        self._actions = []
        self._first_step = True
        self._step = 0
        obs = np.array(self.simulator.reset(self._init_state))

        if not self.blackbox_sim_state:
            obs = np.concatenate((obs, np.array(self._init_state)), axis=0)

        self.root_action = self._action

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

    def get_cache_list(self):
        """Returns the environment info cache.

        Returns
        -------
        dict
            A dictionary containing diagnostic and logging information for the environment.
        """
        return self._info

    def log(self):
        r"""Calls the simulator's `log` function.

        """
        self.simulator.log()

    def render(self, **kwargs):
        r"""Calls the simulator's `render` function, if it exists.

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
            self.simulator.close()
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

    def get_params_internal(self, **tags):
        r"""Returns the parameters associated with the given tags.

        Parameters
        ----------
        tags : dict[bool]
            For each tag, a parameter is returned if the parameter name matches the tag's key
        Returns
        -------
        list
            List of parameters
        """
        # this lasagne function also returns all var below the passed layers
        if not self.params_set:
            self.p_db_filename = GoExploreParameter("db_filename", self.db_filename)
            self.p_key_list = GoExploreParameter("key_list", self.key_list)
            self.p_max_value = GoExploreParameter("max_value", self.max_value)
            self.p_robustify_state = GoExploreParameter("robustify_state", self.robustify_state)
            self.params_set = True

        if tags.pop("db_filename", False):
            return [self.p_db_filename]

        if tags.pop("key_list", False):
            return [self.p_key_list]

        if tags.pop("max_value", False):
            return [self.p_max_value]

        if tags.pop("robustify_state", False):
            return [self.p_robustify_state]

        return [self.p_db_filename, self.p_key_list, self.p_max_value, self.p_robustify_state]  # , self.p_downsampler]

    def set_param_values(self, param_values, **tags):
        r"""Set the values of parameters

        Parameters
        ----------
        param_values : object
            Value to set the parameter to.
        tags : dict[bool]
            For each tag, a parameter is returned if the parameter name matches the tag's key
        """
        debug = tags.pop("debug", False)

        for param, value in zip(
                self.get_params(**tags),
                param_values):
            param.set_value(value)
            if debug:
                print("setting value of %s" % param.name)

    def get_param_values(self, **tags):
        """Return the values of internal parameters.

        Parameters
        ----------
        tags : dict[bool]
            For each tag, a parameter is returned if the parameter name matches the tag's key

        Returns
        -------
        list
            A list of parameter values.
        """
        return [
            param.get_value(borrow=True) for param in self.get_params(**tags)
        ]

    def downsample(self, obs):
        """Create a downsampled approximation of the observed simulation state.

        Parameters
        ----------
        obs : array_like
            The observed simulation state.

        Returns
        -------
        array_like
            The downsampled approximation of the observed simulation state.
        """
        return obs


class Custom_GoExploreASTEnv(GoExploreASTEnv):
    r"""Custom class to change how downsampling works.

        Example class of how to overload downsample to make the environment work for different environments.
    """

    def downsample(self, obs, step=None):
        """Create a downsampled approximation of the observed simulation state.

        Parameters
        ----------
        obs : array_like
            The observed simulation state.
        step : int, optional
            The current iteration number

        Returns
        -------
        array_like
            The downsampled approximation of the observed simulation state.
        """
        obs = obs * 1000
        if step is None:
            step = self._step

        return np.concatenate((np.array([step]), obs), axis=0).astype(int)
