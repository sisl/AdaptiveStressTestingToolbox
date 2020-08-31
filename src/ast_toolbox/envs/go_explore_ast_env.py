import pdb
import pickle
import random
import shelve
from contextlib import contextmanager

import gym
import numpy as np
import tensorflow as tf
from bsddb3 import db
from cached_property import cached_property
from garage.envs.base import Step
from garage.envs.env_spec import EnvSpec
from garage.misc.tensor_utils import flatten_tensors
from garage.misc.tensor_utils import unflatten_tensors

from ast_toolbox.rewards import ExampleAVReward
from ast_toolbox.simulators import ExampleAVSimulator
from ast_toolbox.spaces import ExampleAVSpaces

load_params = True


@contextmanager
def suppress_params_loading():
    global load_params
    load_params = False
    yield
    load_params = True


class Parameterized:
    def __init__(self):
        self._cached_params = {}
        self._cached_param_dtypes = {}
        self._cached_param_shapes = {}
        self._cached_assign_ops = {}
        self._cached_assign_placeholders = {}

    def get_params_internal(self, **tags):
        """
        Internal method to be implemented which does not perform caching
        """
        raise NotImplementedError

    def get_params(self, **tags):
        """
        Get the list of parameters, filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(**tags)
        return self._cached_params[tag_tuple]

    def get_param_dtypes(self, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_dtypes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_dtypes[tag_tuple] = [
                val.dtype for val in param_values
            ]
        return self._cached_param_dtypes[tag_tuple]

    def get_param_shapes(self, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_shapes[tag_tuple] = [
                val.shape for val in param_values
            ]
        return self._cached_param_shapes[tag_tuple]

    def get_param_values(self, **tags):
        params = self.get_params(**tags)
        param_values = tf.get_default_session().run(params)
        return flatten_tensors(param_values)

    def set_param_values(self, flattened_params, name=None, **tags):
        with tf.name_scope(name, 'set_param_values', [flattened_params]):
            debug = tags.pop('debug', False)
            param_values = unflatten_tensors(flattened_params,
                                             self.get_param_shapes(**tags))
            ops = []
            feed_dict = dict()
            for param, dtype, value in zip(
                    self.get_params(**tags), self.get_param_dtypes(**tags),
                    param_values):
                if param not in self._cached_assign_ops:
                    assign_placeholder = tf.placeholder(
                        dtype=param.dtype.base_dtype)
                    assign_op = tf.assign(param, assign_placeholder)
                    self._cached_assign_ops[param] = assign_op
                    self._cached_assign_placeholders[
                        param] = assign_placeholder
                ops.append(self._cached_assign_ops[param])
                feed_dict[self._cached_assign_placeholders[
                    param]] = value.astype(dtype)
                if debug:
                    print('setting value of %s' % param.name)
            tf.get_default_session().run(ops, feed_dict=feed_dict)

    def flat_to_params(self, flattened_params, **tags):
        return unflatten_tensors(flattened_params,
                                 self.get_param_shapes(**tags))

    # def __getstate__(self):
    #     d = Serializable.__getstate__(self)
    #     global load_params
    #     if load_params:
    #         d['params'] = self.get_param_values()
    #     return d
    #
    # def __setstate__(self, d):
    #     Serializable.__setstate__(self, d)
    #     global load_params
    #     if load_params:
    #         tf.get_default_session().run(
    #             tf.variables_initializer(self.get_params()))
    #         self.set_param_values(d['params'])


class JointParameterized(Parameterized):
    def __init__(self, components):
        super(JointParameterized, self).__init__()
        self.components = components

    def get_params_internal(self, **tags):
        params = [
            param for comp in self.components
            for param in comp.get_params_internal(**tags)
        ]
        # only return unique parameters
        return sorted(set(params), key=hash)


class GoExploreParameter():
    def __init__(self, name, value, **tags):
        self.name = name
        self.value = value
        # pdb.set_trace()

    def get_value(self, **kwargs):
        return self.value

    def set_value(self, value):
        self.value = value


#
# class GoExploreASTGymEnv(gym.Env):
#
#     def __init__(self, test):
#
#         print(test)
#
#     def step(self, action):
#         pass
#
#     def reset(self):
#         pass
#
#     def render(self, mode='human'):
#         pass

class GoExploreASTEnv(gym.Env, Parameterized):

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
        # Proportional sampling: Stochastic Acceptance
        # https://arxiv.org/pdf/1109.3627.pdf
        # https://jbn.github.io/fast_proportional_selection/
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
        # obs = self.env.env.reset()
        # state = self.env.env.clone_state()
        obs = self.env_reset()
        if self.blackbox_sim_state:
            # obs = self.downsample(self.simulator.get_first_action())
            obs = self.simulator.get_first_action()
            # else:
        #     obs = self.env_reset()
        # obs = self.simulator.reset(self._init_state)
        state = np.concatenate((self.simulator.clone_state(),
                                np.array([self._cum_reward]),
                                np.array([-1])),
                               axis=0)
        # pdb.set_trace()
        return obs, state

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
            # print('Open Loop:', obs)
            obs = np.array(self._init_state)
            # if not self.robustify:
            # action_return = self.downsample(action_return)
        # if self.simulator.is_goal():
        # Add step number to differentiate identical actions
        # obs = np.concatenate((np.array([self._step]), self.downsample(obs)), axis=0)
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

        # if self.robustify:
        #     # No obs?
        #     # pdb.set_trace()
        #     obs = self._init_state
        # else:
        #     # print(self.robustify_state)
        #     obs = self.downsample(obs)

        # pdb.set_trace()

        return Step(observation=obs,
                    reward=self._reward,
                    done=self._done,
                    cache=self._info,
                    actions=action_return,
                    # step = self._step -1,
                    # real_actions=self._action,
                    state=self._env_state_before_action,
                    root_action=self.root_action,
                    is_terminal=self.simulator.is_terminal(),
                    is_goal=self.simulator.is_goal())

    def simulate(self, actions):
        if not self._fixed_init_state:
            self._init_state = self.observation_space.sample()
        return self.simulator.simulate(actions, self._init_state)

    def reset(self, **kwargs):
        """
        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Calls reset on wrapped env.
        """

        try:
            # print(self.p_robustify_state.value)
            if self.p_robustify_state is not None and self.p_robustify_state.value is not None and len(
                    self.p_robustify_state.value) > 0:
                state = self.p_robustify_state.value
                # print('-----------Robustify Init-----------------')
                # print('-----------Robustify Init: ', state, ' -----------------')
                self.simulator.restore_state(state[:-2])
                obs = self.simulator._get_obs()
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
            # pdb.set_trace()
            # start = time.time()
            flag = db.DB_RDONLY
            pool_DB = db.DB()
            # tick1 = time.time()
            pool_DB.open(self.p_db_filename.value, dbname=None, dbtype=db.DB_HASH, flags=flag)
            # tick2 = time.time()
            dd_pool = shelve.Shelf(pool_DB, protocol=pickle.HIGHEST_PROTOCOL)
            # tick3 = time.time()
            # keys = dd_pool.keys()
            # tick4_1 = time.time()
            # list_of_keys = list(keys)
            # tick4_2 = time.time()
            # choice = random.choice(self.p_key_list.value)
            # import pdb; pdb.set_trace()
            # tick4_3 = time.time()
            # cell = dd_pool[choice]
            cell = self.sample(dd_pool)
            # tick5 = time.time()
            dd_pool.close()
            # tick6 = time.time()
            pool_DB.close()
            # tick7 = time.time()
            # print("Make DB: ", 100*(tick1 - start)/(tick7 - start), " %")
            # print("Open DB: ", 100*(tick2 - tick1) / (tick7 - start), " %")
            # print("Open Shelf: ", 100*(tick4_2 - tick2) / (tick7 - start), " %")
            # # print("Get all keys: ", 100*(tick4_1 - tick3) / (tick7 - start), " %")
            # # print("Make list of all keys: ", 100 * (tick4_2 - tick4_1) / (tick7 - start), " %")
            # print("Choose random cell: ", 100 * (tick4_3 - tick4_2) / (tick7 - start), " %")
            # print("Get random cell: ", 100*(tick5 - tick4_3) / (tick7 - start), " %")
            # print("Close shelf: ", 100*(tick6 - tick5) / (tick7 - start), " %")
            # print("Close DB: ", 100*(tick7 - tick6) / (tick7 - start), " %")
            # print("DB Access took: ", time.time() - start, " s")
            if cell.state is not None:
                # pdb.set_trace()
                if np.all(cell.state == 0):
                    print("-------DEFORMED CELL STATE-------")
                    obs = self.env_reset()
                else:
                    # print("restore state: ", cell.state)
                    self.simulator.restore_state(cell.state[:-2])
                    if self.simulator.is_terminal() or self.simulator.is_goal():
                        print('-------SAMPLED TERMINAL STATE-------')
                        pdb.set_trace()
                        obs = self.env_reset()

                    else:
                        # print("restored")
                        if cell.score == 0.0 and cell.parent is not None:
                            print("Reset to cell with score 0.0 ---- terminal: ", self.simulator.is_terminal(),
                                  " goal: ", self.simulator.is_goal(), " obs: ", cell.observation)
                        obs = self.simulator._get_obs()
                        self._done = False
                        self._cum_reward = cell.state[-2]
                        self._step = cell.state[-1]
                        self.root_action = cell.observation
                    # print("restore obs: ", obs)
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
        self._info = {'actions': []}
        self._action = self.simulator.get_first_action()
        self._actions = []
        self._first_step = True
        self._step = 0
        obs = np.array(self.simulator.reset(self._init_state))
        # if self.blackbox_sim_state:
        #     obs = np.array([0] * self.action_space.shape[0])
        # else:
        #     print('Not action only')
        if not self.blackbox_sim_state:
            obs = np.concatenate((obs, np.array(self._init_state)), axis=0)

        # self.root_action = self.downsample(self._action)
        self.root_action = self._action

        # obs = np.concatenate((np.array([self._step]), self.downsample(obs)), axis=0)
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

    def render(self, **kwargs):
        if hasattr(self.simulator, "render") and callable(getattr(self.simulator, "render")):
            return self.simulator.render(**kwargs)
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

    def get_params_internal(self, **tags):
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
        debug = tags.pop("debug", False)

        for param, value in zip(
                self.get_params(**tags),
                param_values):
            param.set_value(value)
            if debug:
                print("setting value of %s" % param.name)

    def get_param_values(self, **tags):
        return [
            param.get_value(borrow=True) for param in self.get_params(**tags)
        ]

    def downsample(self, obs):
        return obs

    def _get_obs(self):
        return self.simulator._get_obs()


class Custom_GoExploreASTEnv(GoExploreASTEnv):
    def downsample(self, obs, step=None):
        obs = obs * 1000
        if step is None:
            step = self._step

        return np.concatenate((np.array([step]), obs), axis=0).astype(int)
