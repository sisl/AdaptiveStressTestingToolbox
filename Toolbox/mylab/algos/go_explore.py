"""Natural Policy Gradient Optimization."""
from enum import Enum, unique

import numpy as np
import tensorflow as tf

# from garage.logger import logger, tabular
from garage.misc import special
from garage.misc.overrides import overrides
from garage.tf.algos.batch_polopt import BatchPolopt
from garage.tf.misc import tensor_utils
from garage.tf.misc.tensor_utils import compute_advantages
from garage.tf.misc.tensor_utils import discounted_returns
from garage.tf.misc.tensor_utils import filter_valids
from garage.tf.misc.tensor_utils import filter_valids_dict
from garage.tf.misc.tensor_utils import flatten_batch
from garage.tf.misc.tensor_utils import flatten_batch_dict
from garage.tf.misc.tensor_utils import flatten_inputs
from garage.tf.misc.tensor_utils import graph_inputs
from garage.tf.optimizers import LbfgsOptimizer
from cached_property import cached_property
from dowel import logger, tabular
from mylab.envs.go_explore_atari_env import GoExploreTfEnv #, CellPool,Cell
import sys
import pdb
import time
# import xxhash
from bsddb3 import db
import pickle
import shelve
import os

class CellPool():
    def __init__(self, filename = 'database.dat', flag=db.DB_RDONLY, flag2='r'):
        # print("Creating new Cell Pool:", self)
        # self.guide = set()

        # import pdb; pdb.set_trace()
        # self.pool = [self.init_cell]
        # self.guide = self.init_cell.observation
        self.length = 0

        # self.d_pool = {}

        pool_DB = db.DB()
        # print('Creating Cell Pool with flag:', flag)
        # print(filename)
        pool_DB.open(filename, dbname=None, dbtype=db.DB_HASH, flags=flag)
        # pool_DB = None
        # self.d_pool = shelve.Shelf(pool_DB, protocol=pickle.HIGHEST_PROTOCOL)
        self.key_list = []
        self.max_value = 0
        self.max_score = -np.inf
        self.max_reward = -np.inf
        self.best_cell = None
        # self.d_pool = shelve.BsdDbShelf(pool_DB)
        # self.d_pool = shelve.open('/home/mkoren/Scratch/cellpool-shelf2', flag=flag2)
        # self.d_pool = shelve.DbfilenameShelf('/home/mkoren/Scratch/cellpool-shelf2', flag=flag2)

    def create(self, d_pool):

        self.init_cell = Cell()
        self.init_cell.observation = np.zeros((1,128))
        self.init_cell.trajectory = None
        self.init_cell.score = -np.inf
        self.init_cell.reward = -np.inf
        self.init_cell.state = None
        self.init_cell.times_chosen = 0
        self.init_cell.times_visited = 1
        # self.d_pool = shelve.open('cellpool-shelf', flag=flag)

        d_pool[str(hash(self.init_cell))] = self.init_cell
        self.key_list.append(str(hash(self.init_cell)))
        self.length = 1
        self.max_value = self.init_cell.fitness
        # import pdb; pdb.set_trace()

    # def append(self, cell):
    #     # pdb.set_trace()
    #     # if observation not in self.guide:
    #     #     self.guide.add(observation)
    #     #     cell = Cell()
    #     #     cell.observation = observation
    #     #     self.pool.append(cell)
    #     #     self.length += 1
    #     if cell in self.d_pool:
    #         self.d_pool[cell].seen += 1
    #     else:
    #         self.d_pool[cell] = cell


    # def get_cell(self, index):
    #     return self.pool[index]
    #
    # def get_random_cell(self):
    #     index = np.random.randint(0, self.length)
    #     return self.get_cell(index)

    def d_update(self, d_pool, observation, trajectory, score, state, reward=-np.inf, chosen=0):
        # pdb.set_trace()
        #This tests to see if the observation is already in the matrix
        obs_hash = str(hash(observation.tostring()))
        if not obs_hash in d_pool:
            # self.guide.add(observation)
            cell = Cell()
            cell.observation = observation
            # self.guide = np.append(self.guide, np.expand_dims(observation, axis=0), axis = 0)
            cell.trajectory = trajectory
            cell.score = score
            cell.trajectory_length = len(trajectory)
            cell.state = state
            cell.times_visited = 1
            cell.times_chosen = chosen
            cell.times_chosen_since_improved = 0
            cell.reward = reward
            d_pool[obs_hash] = cell
            self.length += 1
            self.key_list.append(obs_hash)
            if cell.fitness > self.max_value:
                self.max_value = cell.fitness
            if cell.score > self.max_score:
                self.max_score = score

            return True
        else:
            cell = d_pool[obs_hash]
            if score > cell.score:
                cell.score = score
                cell.trajectory = trajectory
                cell.trajectory_length = len(trajectory)
                cell.state = state
                cell.reward = reward

            cell.times_visited += 1
            cell.times_chosen += chosen
            d_pool[obs_hash] = cell
            if cell.fitness > self.max_value:
                self.max_value = cell.fitness
            if cell.score > self.max_score:
                self.max_score = score
        return False


class Cell():

    def __init__(self):
        # print("Creating new Cell:", self)
        # Number of times this was chosen and seen
        self._times_visited=0
        self._times_chosen = 0
        self._times_chosen_since_improved = 0
        self._score = -np.inf
        self._reward = 0
        self._action_times = 0

        self.trajectory_length = -np.inf
        self.trajectory = np.array([])
        self.state = None
        self.observation = None
        # self._is_root = False

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        if np.all(self.observation == other.observation):
            return True
        else:
            return False

    def reset_cached_property(self, cached_property):
        if cached_property in self.__dict__:
            del self.__dict__[cached_property]

    @property
    def is_root(self):
        return len(self.trajectory) == 0

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, value):
        self._reward = value
        self.reset_cached_property('score_weight')
        self.reset_cached_property('fitness')

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value
        self.reset_cached_property('score_weight')
        self.reset_cached_property('fitness')

    @property
    def times_visited(self):
        return self._times_visited

    @times_visited.setter
    def times_visited(self, value):
        self._times_visited = value
        self.reset_cached_property('times_visited_subscore')
        self.reset_cached_property('count_subscores')
        self.reset_cached_property('fitness')

    @property
    def times_chosen(self):
        return self._times_chosen

    @times_chosen.setter
    def times_chosen(self, value):
        self._times_chosen = value
        self.reset_cached_property('times_chosen_subscore')
        self.reset_cached_property('count_subscores')
        self.reset_cached_property('fitness')

    @property
    def times_chosen_since_improved(self):
        return self._times_chosen_since_improved

    @times_chosen_since_improved.setter
    def times_chosen_since_improved(self, value):
        self._times_chosen_since_improved = value
        self.reset_cached_property('times_chosen_since_improved')
        self.reset_cached_property('count_subscores')
        self.reset_cached_property('fitness')


    @cached_property
    def fitness(self):
        # return max(1, self.score)
        return self.score_weight*(self.count_subscores + 1)

    @cached_property
    def count_subscores(self):
        return (self.times_chosen_subscore +
                self.times_chosen_since_improved_subscore +
                self.times_visited_subscore)

    @cached_property
    def times_chosen_subscore(self):
        weight = 0.1
        power = 0.5
        eps1 = 0.001
        eps2 = 0.00001
        return weight * (1 / (self.times_chosen + eps1)) ** power + eps2

    @cached_property
    def times_chosen_since_improved_subscore(self):
        weight = 0.0
        power = 0.5
        eps1 = 0.001
        eps2 = 0.00001
        return weight * (1 / (self.times_chosen_since_improved + eps1)) ** power + eps2

    @cached_property
    def times_visited_subscore(self):
        weight = 0.3
        power = 0.5
        eps1 = 0.001
        eps2 = 0.00001
        return weight * (1 / (self.times_chosen + eps1)) ** power + eps2

    @cached_property
    def score_weight(self):
        return 1.0
        # return min(1e-6, 0.1**max(0.0, (100000-self.score)/10000))

    def __hash__(self):
        return hash((self.observation.tostring()))

class GoExplore(BatchPolopt):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo,
    etc.
    """

    def __init__(self,
                 db_filename,
                 max_db_size,
                 env,
                 env_spec,
                 policy,
                 baseline,
                 **kwargs):
        # env,
        # policy,
        # baseline,
        # scope = None,
        # n_itr = +500,
        # start_itr = 0,
        # batch_size = 5000,
        # max_path_length = 500,
        # discount = 0.99,
        # gae_lambda = 1,
        # plot = False,
        # pause_for_plot = False,
        # center_adv = True,
        # positive_adv = False,
        # store_paths = False,
        # whole_paths = True,
        # fixed_horizon = False,
        # sampler_cls = None,
        # sampler_args = None,
        # force_batch_sampler = False,
        """
        :param env_spec: Environment specification.
        :type env_spec: EnvSpec
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if
         running multiple algorithms
        simultaneously, each using different environments and policies
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param center_adv: Whether to rescale the advantages so that they have
         mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are
         always positive. When used in conjunction with center_adv the
         advantages will be standardized before shifting.
        :return:
        """
        self.db_filename = db_filename
        self.max_db_size = max_db_size
        self.env_spec = env_spec
        self.policy = policy
        self.env = env
        self.best_cell = None

        # self.init_opt()

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         **kwargs)

    @overrides
    def train(self, runner, batch_size):
        last_return = None

        for epoch in runner.step_epochs():
            runner.step_path = runner.obtain_samples(runner.step_itr,
                                                     batch_size)
            last_return = self.train_once(runner.step_itr, runner.step_path)
            runner.step_itr += 1

        return last_return

    @overrides
    def train_once(self, itr, paths):
        paths = self.process_samples(itr, paths)

        self.log_diagnostics(paths)
        logger.log('Optimizing policy...')
        self.optimize_policy(itr, paths)
        return self.best_cell

    @overrides
    def init_opt(self):
        self.max_cum_reward = -np.inf
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        # pdb.set_trace()
        # self.temp_index = 0
        # pool_DB = db.DB()
        # pool_DB.open(self.db_filename, dbname=None, dbtype=db.DB_HASH, flags=db.DB_CREATE)
        self.cell_pool = CellPool(filename=self.db_filename, flag=db.DB_CREATE, flag2='n')
        # self.cell_pool.create()
        # obs = self.env.downsample(self.env.env.env.reset())
        pool_DB = db.DB()
        pool_DB.open(self.db_filename, dbname=None, dbtype=db.DB_HASH, flags=db.DB_CREATE)
        d_pool = shelve.Shelf(pool_DB, protocol=pickle.HIGHEST_PROTOCOL)
        obs, state = self.env.get_first_cell()
        # pdb.set_trace()
        self.cell_pool.d_update(d_pool=d_pool, observation=obs, trajectory=np.array([]), score=0.0, state=state, reward=0.0, chosen=1)
        d_pool.sync()
        # self.cell_pool.d_pool.close()
        # cell = Cell()
        # cell.observation = np.zeros(128)
        # self.temp_index += 1
        # self.cell_pool.append(cell)
        # self.cell_pool.update(observation=np.zeros(128), trajectory=None, score=-np.infty, state=None)
        self.env.set_param_values([self.db_filename], db_filename=True, debug=False)
        self.env.set_param_values([self.cell_pool.key_list], key_list=True, debug=False)
        self.env.set_param_values([self.cell_pool.max_value], max_value=True, debug=False)
        d_pool.close()
        # pdb.set_trace()
        # self.policy.set_param_values({"cell_num":-1,
        #                               "stateful_num":-1,
        #                               "cell_pool": self.cell_pool})
        # self.policy.set_cell_pool(self.cell_pool)
        # self.env.set_cell_pool(self.cell_pool)
        # GoExploreTfEnv.pool.append(Cell())
        # self.env.append_cell(Cell())
        # self.env.set_param_values(self.env.pool, pool=True)
        # self.env.set_param_values([np.random.randint(0,100)], debug=True,test_var=True)


    @overrides
    def get_itr_snapshot(self, itr):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        pdb.set_trace()
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
        )

    @overrides
    def optimize_policy(self, itr, samples_data):

        start = time.time()

        pool_DB = db.DB()
        pool_DB.open(self.db_filename, dbname=None, dbtype=db.DB_HASH, flags=db.DB_CREATE)
        d_pool = shelve.Shelf(pool_DB, protocol=pickle.HIGHEST_PROTOCOL)

        new_cells = 0
        total_cells = 0
        # self.cell_pool.d_pool.open()
        print("(2) Processing Samples...")
        # pdb.set_trace()
        for i in range(samples_data['observations'].shape[0]):
            sys.stdout.write("\rProcessing Trajectory {0} / {1}".format(i, samples_data['observations'].shape[0]))
            sys.stdout.flush()
            cum_reward = 0
            cum_traj = np.array([])
            observation = None
            for j in range(samples_data['observations'].shape[1]):
                # pdb.set_trace()
                chosen = 0
                if j == 0:
                    chosen = 1
                    try:
                        root_cell = d_pool[str(hash(samples_data['observations'][i, j, :].tostring()))]
                        cum_reward = root_cell.reward
                        cum_traj = root_cell.trajectory
                    except:
                        pdb.set_trace()
                    # if cum_reward == 0 or cum_reward  <-1e8:
                    #     pdb.set_trace()

                if np.all(samples_data['observations'][i, j, :] == 0):
                    continue
                observation = samples_data['observations'][i, j, :]
                trajectory = samples_data['observations'][i, 0:j, :]
                if cum_traj.shape[0] > 0:
                    trajectory = np.concatenate((cum_traj, trajectory), axis=0)
                score = samples_data['rewards'][i, j]
                cum_reward += score
                state = samples_data['env_infos']['state'][i, j, :]
                if self.cell_pool.d_update(d_pool=d_pool,
                                           observation=observation,
                                           trajectory=trajectory,
                                           score=score,
                                           state=state,
                                           reward=cum_reward,
                                           chosen=chosen):
                    new_cells += 1
                total_cells += 1
            if cum_reward > self.max_cum_reward and observation is not None:
                self.max_cum_reward = cum_reward
                self.best_cell = d_pool[str(hash(observation.tostring()))]
            if cum_reward > -100:
                pdb.set_trace()
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(new_cells, " new cells (", 100 * new_cells / total_cells, "%)")
        print(total_cells, " samples processed in ", time.time() - start, " seconds")

        d_pool.sync()
        d_pool.close()
        # self.cell_pool.d_pool.close()
        # self.env.set_param_values([self.cell_pool], pool=True, debug=True)
        self.env.set_param_values([self.cell_pool.key_list], key_list=True, debug=False)
        self.env.set_param_values([self.cell_pool.max_value], max_value=True, debug=False)

        if os.path.getsize(self.db_filename) /1000/1000/1000 > self.max_db_size:
            print ('------------ERROR: MAX DB SIZE REACHED------------')
            sys.exit()
        print('\n---------- Max Score: ', self.max_cum_reward, ' ----------------\n')

