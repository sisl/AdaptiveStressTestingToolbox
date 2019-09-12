from garage.tf.envs.base import TfEnv
import numpy as np
from multiprocessing import Value
from garage.misc.overrides import overrides
import pdb
import random
import shelve
from bsddb3 import db
import pickle
import time
from cached_property import cached_property
from garage.core import Parameterized

class GoExploreParameter():
    def __init__(self, name, value, **tags):
        self.name = name
        self.value = value
        # pdb.set_trace()

    def get_value(self, **kwargs):
        return self.value

    def set_value(self, value):
        self.value = value




class GoExploreTfEnv(TfEnv, Parameterized):

    def __init__(self, env=None, env_name="", simulator = None):
        self.params_set = False
        self.db_filename = 'database.dat'
        self.key_list = []
        self.max_value = 0
        self.wrapped_env = env
        if simulator = None:
            self.simulator = self.env.env
        else:
            self.simulator = simulator

        super().__init__(env, env_name)
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
        obs = self.simulator.reset()
        state = self.simulator.clone_state()
        return obs, state

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
                    obs = self.env.env.reset()
                else:
                    # print("restore state: ", cell.state)
                    self.env.env.restore_state(cell.state)
                    # print("restored")
                    obs = self.env.env._get_obs()
                    # print("restore obs: ", obs)
            else:
                print("Reset from start")
                obs = self.env.env.reset()
            # pdb.set_trace()
        except db.DBBusyError:
            print("DBBusyError")
            obs = self.env.env.reset()
        except db.DBLockNotGrantedError or db.DBLockDeadlockError:
            print("db.DBLockNotGrantedError or db.DBLockDeadlockError")
            obs = self.env.env.reset()
        except db.DBForeignConflictError:
            print("DBForeignConflictError")
            obs = self.env.env.reset()
        except db.DBAccessError:
            print("DBAccessError")
            obs = self.env.env.reset()
        except db.DBPermissionsError:
            print("DBPermissionsError")
            obs = self.env.env.reset()
        except db.DBNoSuchFileError:
            print("DBNoSuchFileError")
            obs = self.env.env.reset()
        except db.DBError:
            print("DBError")
            obs = self.env.env.reset()
        except:
            print("Failed to get state from database")
            obs = self.env.env.reset()


        x = self.downsample(obs)
        return x
    #
    def step(self, action):
        """
        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Calls step on wrapped env.
        """

        try:
            obs, reward, done, env_info = self.env.env.step(action)
            env_info['state'] = self.env.env.clone_state()
            if env_info['state'][0] == 0:
                print("GOT DEFORMED STATE: ", obs, reward, action, done)
                import sys; sys.exit()
            return self.downsample(obs), reward, done, env_info
        except:
            pdb.set_trace()


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



# Might be able to speed up by having class of (observation, index) where hash and eq
# are observation, but getting obs also gets us the index
