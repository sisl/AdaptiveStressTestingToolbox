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
from mylab.envs.go_explore_env import GoExploreTfEnv, CellPool,Cell
import sys
import pdb
import time
# import xxhash
from bsddb3 import db
import os


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

        # self.init_opt()

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         **kwargs)

    @overrides
    def init_opt(self):
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
        obs, state = self.env.get_first_cell()
        self.cell_pool.d_update(observation=obs, trajectory=[], score=0.0, state=state, chosen=1)
        self.cell_pool.d_pool.sync()
        # self.cell_pool.d_pool.close()
        # cell = Cell()
        # cell.observation = np.zeros(128)
        # self.temp_index += 1
        # self.cell_pool.append(cell)
        # self.cell_pool.update(observation=np.zeros(128), trajectory=None, score=-np.infty, state=None)
        self.env.set_param_values([self.db_filename], db_filename=True, debug=False)
        self.env.set_param_values([self.cell_pool.key_list], key_list=True, debug=False)
        self.env.set_param_values([self.cell_pool.max_value], max_value=True, debug=False)
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
    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        return {'env':None, 'paths':None}

    @overrides
    def optimize_policy(self, itr, samples_data):

        start = time.time()
        new_cells = 0
        total_cells = 0
        # self.cell_pool.d_pool.open()
        print("(2) Processing Samples...")
        # pdb.set_trace()
        for i in range(samples_data['observations'].shape[0]):
            sys.stdout.write("\rProcessing Trajectory {0} / {1}".format(i, samples_data['observations'].shape[0]))
            sys.stdout.flush()
            for j in range(samples_data['observations'].shape[1]):
                # pdb.set_trace()
                observation = samples_data['observations'][i, j, :] #// 32
                trajectory = samples_data['observations'][i, 0:j, :]
                score = samples_data['rewards'][i, j]
                state = samples_data['env_infos']['state'][i, j, :]
                if self.cell_pool.d_update(observation, trajectory, score, state):
                    new_cells += 1
                total_cells += 1
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(new_cells, " new cells (", 100 * new_cells / total_cells, "%)")
        print(total_cells, " samples processed in ", time.time() - start, " seconds")
        self.cell_pool.d_pool.sync()
        # self.cell_pool.d_pool.close()
        #TODO Way too much memory having to copy the whole pool, need to just set the single cell if possible
        # self.env.set_param_values([self.cell_pool], pool=True, debug=True)
        self.env.set_param_values([self.cell_pool.key_list], key_list=True, debug=False)
        self.env.set_param_values([self.cell_pool.max_value], max_value=True, debug=False)
        if os.path.getsize(self.db_filename) /1000/1000/1000 > self.max_db_size:
            print ('------------ERROR: MAX DB SIZE REACHED------------')
            sys.exit()
        print('\n---------- Max Score: ', self.cell_pool.max_score, ' ----------------\n')

# class CellPool():
#     def __init__(self):
#         print("Creating new Cell Pool:", self)
#         self.pool = []
#         self.length = 0
#
#     def append(self, cell):
#         self.pool.append(cell)
#         self.length += 1
#
#     def get_cell(self, index):
#         return self.pool[index]
#
# class Cell():
#
#     def __init__(self):
#         print("Creating new Cell:", self)
#         # Number of times this was chosen and seen
#         self.seen_times = []
#         self.chosen_times = 0
#         self.chosen_since_new = 0
#         self.score = -np.inf
#         self.action_times = 0
#
#         self.trajectory_length = -np.inf
#         self.trajectory = []
#
#     def __eq__(self, other):
#         if type(other) != type(self):
#             return False
#         # if self.trajectory
#
#     # @dataclass
#     # class Cell:
#     #     # The list of ChainLink that can take us to this place
#     #     chain: typing.List[ChainLink] = copyfield([])
#     #     seen: list = copyfield({})
#     #     score: int = -infinity
#     #
#     #     seen_times: int = 0
#     #     chosen_times: int = 0
#     #     chosen_since_new: int = 0
#     #     action_times: int = 0  # This is the number of action that led to this cell
#     #     # Length of the trajectory
#     #     trajectory_len: int = infinity
#     #     # Saved restore state. In a purely deterministic environment,
#     #     # this allows us to fast-forward to the end state instead
#     #     # of replaying.
#     #     restore: typing.Any = None
#     #     # TODO: JH: This should not refer to a Montezuma-only data-structure
#     #     exact_pos: MontezumaPosLevel = None
#     #     trajectory: list = copyfield([])
#     #     real_cell: MontezumaPosLevel = None
