from cached_property import cached_property
from garage.misc.overrides import overrides

from garage.envs.base import Step


import numpy as np
from mylab.simulators.example_av_simulator import ExampleAVSimulator
from mylab.rewards.example_av_reward import ExampleAVReward
from garage.envs.env_spec import EnvSpec
import pdb
import gym
from garage.core import Serializable, Parameterized
import random
import shelve
from bsddb3 import db
import pickle

class GoExploreParameter():
    def __init__(self, name, value, **tags):
        self.name = name
        self.value = value
        # pdb.set_trace()

    def get_value(self, **kwargs):
        return self.value

    def set_value(self, value):
        self.value = value

class GoExploreASTGymEnv(gym.Env):

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

class GoExploreASTEnv(gym.Wrapper, Parameterized):
# class ASTEnv(GarageEnv):
    def __init__(self,
                 open_loop=True,
                 action_only=True,
                 fixed_init_state=False,
                 s_0=None,
                 simulator=None,
                 reward_function=None,
                 spaces=None):
        super().__init__(gym.make('mylab:GoExploreAST-v0'))
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
        # Always call Serializable constructor last
        self.params_set = False
        self.db_filename = 'database.dat'
        self.key_list = []
        self.max_value = 0

        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)

    def sample(self, population):
        #Proportional sampling: Stochastic Acceptance
        # https://arxiv.org/pdf/1109.3627.pdf
        # https://jbn.github.io/fast_proportional_selection/
        attempts = 0
        while attempts < 10000:
            candidate = population[random.choice(self.p_key_list.value)]
            if random.random() < (candidate.fitness / self.p_max_value.value):
                return candidate
        print("Returning Uniform Random Sample - Max Attempts Reached!")
        return population[random.choice(self.p_key_list.value)]

    def get_first_cell(self):
        # obs = self.env.env.reset()
        # state = self.env.env.clone_state()
        obs = self.simulator.reset(self._init_state)
        state = self.simulator.clone_state()
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
        self._action = action
        self._actions.append(action)
        # Update simulation step
        obs = self.simulator.step(self._action, self.open_loop)
        if (obs is None) or (self.open_loop is True):
            obs = self._init_state
        # if self.simulator.is_goal():
        if self.simulator.isterminal():
            self._done = True
        # Calculate the reward for this step
        self._reward = self.reward_function.give_reward(
            action=self._action,
            info=self.simulator.get_reward_info())
        # Update instance attributes
        self._step = self._step + 1
        self._simulator_state = self.simulator.clone_state()

        return Step(observation=self.downsample(obs),
                    reward=self._reward,
                    done=self._done,
                    info={'cache': self._info,
                          'actions': self._action,
                          'state': self._simulator_state})

    def simulate(self, actions):
        if not self._fixed_init_state:
            self._init_state = self.observation_space.sample()
        self.simulator.simulate(actions, self._init_state)

    def reset(self, **kwargs):
        """
        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Calls reset on wrapped env.
        """

        try:
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
                if cell.state[0] == 0:
                    print("DEFORMED CELL STATE")
                    obs = self.env_reset()
                else:
                    # print("restore state: ", cell.state)
                    self.simulator.restore_state(cell.state)
                    # print("restored")
                    obs = self.simulator._get_obs()
                    # print("restore obs: ", obs)
            else:
                print("Reset from start")
                obs = self.env_reset()
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
        except:
            print("Failed to get state from database")
            obs = self.env_reset()


        x = self.downsample(obs)
        return x

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
        self._info = []
        self._action = None
        self._actions = []
        self._first_step = True
        self._step = 0

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

    @overrides
    def get_params_internal(self, **tags):
        # this lasagne function also returns all var below the passed layers
        if not self.params_set:
            self.p_db_filename = GoExploreParameter("db_filename", self.db_filename)
            self.p_key_list = GoExploreParameter("key_list", self.key_list)
            self.p_max_value = GoExploreParameter("max_value", self.max_value)
            self.params_set = True

        if tags.pop("db_filename", False) == True:
            return [self.p_db_filename]

        if tags.pop("key_list", False) == True:
            return [self.p_key_list]

        if tags.pop("max_value", False) == True:
            return [self.p_max_value]


        return [self.p_db_filename, self.p_key_list, self.p_max_value]#, self.p_downsampler]

    @overrides
    def set_param_values(self, param_values, **tags):

        debug = tags.pop("debug", False)

        for param,  value in zip(
            self.get_params(**tags),
            param_values):
            param.set_value(value)
            if debug:
                print("setting value of %s" % param.name)

    @overrides
    def get_param_values(self, **tags):
        return [
            param.get_value(borrow=True) for param in self.get_params(**tags)
        ]

    def downsample(self, obs):
        return obs