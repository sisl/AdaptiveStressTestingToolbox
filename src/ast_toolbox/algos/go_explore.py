"""Natural Policy Gradient Optimization."""
import contextlib
import os
import pdb
import pickle
import shelve
import sys
import time

import numpy as np
from bsddb3 import db
from cached_property import cached_property
from dowel import logger
from dowel import tabular
from garage.tf.algos.batch_polopt import BatchPolopt


class Cell():

    def __init__(self, use_score_weight=True):
        # print("Creating new Cell:", self)
        # Number of times this was chosen and seen
        self._times_visited = 0
        self._times_chosen = 0
        self._times_chosen_since_improved = 0
        self._score = -np.inf
        self._reward = 0
        self._value_approx = 0.0
        self._action_times = 0

        self.trajectory_length = -np.inf
        self.trajectory = np.array([])
        self.state = None
        self.observation = None
        self.action = None
        self.parent = None
        self._is_goal = False
        self._is_terminal = False
        # self._is_root = False

        self.use_score_weight = use_score_weight

    def __eq__(self, other):
        if not isinstance(other, type(self)):
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
    def step(self):
        return len(self.trajectory)

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, value):
        self._reward = value
        self.reset_cached_property('score_weight')
        self.reset_cached_property('fitness')

    @property
    def value_approx(self):
        return self._value_approx

    @value_approx.setter
    def value_approx(self, value):
        self._value_approx = value
        self.reset_cached_property('score_weight')
        self.reset_cached_property('fitness')

    @property
    def is_terminal(self):
        return self._is_terminal

    @is_terminal.setter
    def is_terminal(self, value):
        self._is_terminal = value
        self.reset_cached_property('score_weight')
        self.reset_cached_property('fitness')

    @property
    def is_goal(self):
        return self._is_goal

    @is_goal.setter
    def is_goal(self, value):
        self._is_goal = value
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
        return self.score_weight * (self.count_subscores + 1)

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
        if self.use_score_weight:
            score_weight = 1 / max([abs(self._value_approx), 1])
        else:
            score_weight = 1.0  # Not sampling based on score right now
        # Set chance of sampling to 0 if this cell is a terminal state
        terminal_sample_elimination_factor = not(self.is_terminal or self._is_goal)
        return terminal_sample_elimination_factor * score_weight
        # return min(1e-6, 0.1**max(0.0, (100000-self.score)/10000))

    def __hash__(self):
        return hash((self.observation.tostring()))


class CellPool():
    def __init__(self, filename='database', discount=0.99, flag=db.DB_RDONLY, flag2='r', use_score_weight=True):
        # print("Creating new Cell Pool:", self)
        # self.guide = set()

        # import pdb; pdb.set_trace()
        # self.pool = [self.init_cell]
        # self.guide = self.init_cell.observation
        self.length = 0
        self._filename = filename
        self.discount = discount

        # self.d_pool = {}

        # pool_DB = db.DB()
        # print('Creating Cell Pool with flag:', flag)
        # print(filename)
        # pool_DB.open(filename + '_pool.dat', dbname=None, dbtype=db.DB_HASH, flags=flag)
        # pool_DB = None
        # self.d_pool = shelve.Shelf(pool_DB, protocol=pickle.HIGHEST_PROTOCOL)
        self.key_list = []
        self.goal_dict = {}
        self.terminal_dict = {}
        self.max_value = 0
        self.max_score = -np.inf
        self.max_reward = -np.inf
        self.best_cell = None

        self.use_score_weight = use_score_weight

        # self.d_pool = shelve.BsdDbShelf(pool_DB)
        # self.d_pool = shelve.open('/home/mkoren/Scratch/cellpool-shelf2', flag=flag2)
        # self.d_pool = shelve.DbfilenameShelf('/home/mkoren/Scratch/cellpool-shelf2', flag=flag2)

    def save(self):
        best_cell_key = None
        if self.best_cell is not None:
            best_cell_key = str(hash(self.best_cell.observation.tostring()))
        save_dict = {
            'key_list': self.key_list,
            'goal_dict': self.goal_dict,
            'terminal_dict': self.terminal_dict,
            'max_value': self.max_value,
            'max_score': self.max_score,
            'max_reward': self.max_reward,
            'use_score_weight': self.use_score_weight,
            'best_cell': best_cell_key,
        }
        dirname = os.path.dirname(self.meta_filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(self.meta_filename, "wb") as f:
            pickle.dump(save_dict, f)

    def load(self, cell_pool_shelf):
        with contextlib.suppress(FileNotFoundError):
            with open(self.meta_filename, "rb") as f:
                save_dict = pickle.load(f)

                self.key_list = save_dict['key_list']
                self.goal_dict = save_dict['goal_dict']
                self.terminal_dict = save_dict['terminal_dict']
                self.max_value = save_dict['max_value']
                self.max_score = save_dict['max_score']
                self.max_reward = save_dict['max_reward']
                self.use_score_weight = save_dict['use_score_weight']
                self.best_cell = None
                best_cell_key = save_dict['best_cell']
                if best_cell_key is not None:
                    self.best_cell = cell_pool_shelf[best_cell_key]

    def open_pool(self, dbname=None, dbtype=db.DB_HASH, flags=db.DB_CREATE, protocol=pickle.HIGHEST_PROTOCOL, overwrite=False):
        # We can't save our database as a class attribute due to pickling errors.
        # To prevent errors from code repeat, this convenience function opens the database and
        # loads the latest meta data, the returns the database.
        if overwrite:
            self.delete_pool()
        cell_pool_db = db.DB()
        cell_pool_db.open(self.pool_filename, dbname=dbname, dbtype=dbtype, flags=flags)
        cell_pool_shelf = shelve.Shelf(cell_pool_db, protocol=protocol)
        self.load(cell_pool_shelf=cell_pool_shelf)
        return cell_pool_shelf

    def sync_pool(self, cell_pool_shelf):
        # We can't save our database as a class attribute due to pickling errors.
        # To prevent errors from code repeat, this convenience function syncs the given database and
        # saves the latest meta data.
        cell_pool_shelf.sync()
        self.save()

    def close_pool(self, cell_pool_shelf):
        # We can't save our database as a class attribute due to pickling errors.
        # To prevent errors from code repeat, this convenience function closes the given database and
        # saves the latest meta data.
        cell_pool_shelf.close()
        self.save()

    def sync_and_close_pool(self, cell_pool_shelf):
        # We can't save our database as a class attribute due to pickling errors.
        # To prevent errors from code repeat, this convenience function syncs and closes the given
        # database and saves the latest meta data.
        cell_pool_shelf.sync()
        cell_pool_shelf.close()
        self.save()

    def delete_pool(self):
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.pool_filename)
            os.remove(self.meta_filename)

    @cached_property
    def pool_filename(self):
        return self._filename + '_pool.dat'

    @cached_property
    def meta_filename(self):
        return self._filename + '_meta.dat'

    # def create(self, d_pool):
    #
    #     self.init_cell = Cell()
    #     self.init_cell.observation = np.zeros((1,128))
    #     self.init_cell.trajectory = None
    #     self.init_cell.score = -np.inf
    #     self.init_cell.reward = -np.inf
    #     self.init_cell.state = None
    #     self.init_cell.times_chosen = 0
    #     self.init_cell.times_visited = 1
    #     # self.d_pool = shelve.open('cellpool-shelf', flag=flag)
    #
    #     d_pool[str(hash(self.init_cell))] = self.init_cell
    #     self.key_list.append(str(hash(self.init_cell)))
    #     self.length = 1
    #     self.max_value = self.init_cell.fitness
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

    def d_update(self, d_pool, observation, action, trajectory, score, state,
                 parent=None, is_terminal=False, is_goal=False, reward=-np.inf, chosen=0):
        # pdb.set_trace()
        # This tests to see if the observation is already in the matrix

        obs_hash = str(hash(observation.tostring()))
        if obs_hash not in d_pool:
            # Make a new cell, add to pool
            # self.guide.add(observation)
            cell = Cell(self.use_score_weight)
            cell.observation = observation
            # self.guide = np.append(self.guide, np.expand_dims(observation, axis=0), axis = 0)
            cell.action = action
            cell.trajectory = trajectory
            cell.score = score
            cell.trajectory_length = len(trajectory)
            cell.state = state
            cell.times_visited = 1
            cell.times_chosen = chosen
            cell.times_chosen_since_improved = 0
            cell.reward = reward
            cell.parent = parent
            cell.is_terminal = is_terminal
            cell.is_goal = is_goal

            d_pool[obs_hash] = cell
            self.length += 1
            self.key_list.append(obs_hash)
            if cell.fitness > self.max_value:
                self.max_value = cell.fitness
            if cell.score > self.max_score:
                self.max_score = score
            if is_goal:
                self.goal_dict[obs_hash] = cell.reward
            elif is_terminal:
                self.terminal_dict[obs_hash] = cell.reward

            self.value_approx_update(value=cell.value_approx, obs_hash=cell.parent, d_pool=d_pool)

            return True
        else:
            cell = d_pool[obs_hash]
            if score > cell.score:
                # Cell exists, but new version is better. Overwrite
                cell.score = score
                cell.action = action
                cell.trajectory = trajectory
                cell.trajectory_length = len(trajectory)
                cell.state = state
                cell.reward = reward
                cell.parent = parent
                cell.is_terminal = is_terminal
                cell.is_goal = is_goal

                if obs_hash in self.goal_dict:
                    del self.goal_dict[obs_hash]
                if obs_hash in self.terminal_dict:
                    del self.terminal_dict[obs_hash]

                if is_goal:
                    self.goal_dict[obs_hash] = cell.reward
                elif is_terminal:
                    self.terminal_dict[obs_hash] = cell.reward

            cell.times_visited += 1
            cell.times_chosen += chosen
            d_pool[obs_hash] = cell
            if cell.fitness > self.max_value:
                self.max_value = cell.fitness
            if cell.score > self.max_score:
                self.max_score = score

            self.value_approx_update(value=cell.value_approx, obs_hash=cell.parent, d_pool=d_pool)

        return False

    def value_approx_update(self, value, obs_hash, d_pool):
        if obs_hash is not None:
            cell = d_pool[obs_hash]
            v = cell.score + self.discount * value
            cell.value_approx = (v - cell.value_approx) / cell.times_visited + cell.value_approx
            d_pool[obs_hash] = cell
            if cell.parent is not None:
                self.value_approx_update(value=cell.value_approx, obs_hash=cell.parent, d_pool=d_pool)


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
                 # robust_policy,
                 # robust_baseline,
                 # robustify_max,
                 # robustify_algo,
                 # robustify_policy,
                 save_paths_gap=0,
                 save_paths_path=None,
                 overwrite_db=True,
                 use_score_weight=True,
                 **kwargs):

        # algo = TRPO(
        #     env_spec=env.spec,
        #     policy=policy,
        #     baseline=baseline,
        #     max_path_length=max_path_length,
        #     discount=0.99,
        #     kl_constraint='hard',
        #     optimizer=optimizer,
        #     optimizer_args=optimizer_args,
        #     lr_clip_range=1.0,
        #     max_kl_step=1.0)

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
        self.overwrite_db = overwrite_db
        self.max_db_size = max_db_size
        self.env_spec = env_spec
        self.go_explore_policy = policy
        self.use_score_weight = use_score_weight
        # self.robust_policy = robust_policy
        # self.robust_baseline = robust_baseline
        self.env = env
        self.best_cell = None
        self.robustify = False
        # self.robustify_max = robustify_max
        self.save_paths_gap = save_paths_gap
        self.save_paths_path = save_paths_path
        self.policy = self.go_explore_policy

        # self.init_opt()

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         **kwargs)

    def train(self, runner):
        last_return = None
        self.policy = self.go_explore_policy
        for epoch in runner.step_epochs():
            runner.step_path = runner.obtain_samples(runner.step_itr)
            last_return = self.train_once(runner.step_itr, runner.step_path)
            runner.step_itr += 1

        # pdb.set_trace()
        # self.policy = self.robust_policy
        # self.backward_algorithm = BackwardAlgorithm(
        #     env=self.env,
        #     env_spec=self.env_spec,
        #     policy=self.robust_policy,
        #     baseline=self.robust_baseline,
        #     expert_trajectory=last_return.trajectory.tolist(),
        #     epochs_per_step=10)
        # pdb.set_trace()
        # return self.backward_algorithm.train(runner=runner,batch_size=batch_size)

        # self.robustify = True
        # self.init_opt()
        # for epoch in range(self.robustify_max):
        #     runner.step_path = runner.obtain_samples(runner.step_itr,
        #                                              batch_size)
        #     last_return = self.train_once(runner.step_itr, runner.step_path)
        return last_return

    def train_once(self, itr, paths):
        paths = self.process_samples(itr, paths)

        self.log_diagnostics(paths)
        logger.log('Optimizing policy...')
        self.optimize_policy(itr, paths)
        return self.best_cell

    def init_opt(self):

        # if self.robustify:
        #     self.env.set_param_values([None], robustify_state=True, debug=False)
        self.max_cum_reward = -np.inf
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        # pdb.set_trace()
        # self.temp_index = 0
        # pool_DB = db.DB()
        # pool_DB.open(self.db_filename, dbname=None, dbtype=db.DB_HASH, flags=db.DB_CREATE)
        self.cell_pool = CellPool(filename=self.db_filename, use_score_weight=self.use_score_weight)

        # self.cell_pool.create()
        # obs = self.env.downsample(self.env.env.env.reset())
        d_pool = self.cell_pool.open_pool(overwrite=self.overwrite_db)
        if len(self.cell_pool.key_list) == 0:
            obs, state = self.env.get_first_cell()
        # pdb.set_trace()
            self.cell_pool.d_update(d_pool=d_pool, observation=self.downsample(obs, step=-1), action=obs,
                                    trajectory=np.array([]), score=0.0, state=state, reward=0.0, chosen=0)
            self.cell_pool.sync_pool(cell_pool_shelf=d_pool)

        self.max_cum_reward = self.cell_pool.max_reward
        self.best_cell = self.cell_pool.best_cell

        self.cell_pool.close_pool(cell_pool_shelf=d_pool)
        # self.cell_pool.d_pool.close()
        # cell = Cell()
        # cell.observation = np.zeros(128)
        # self.temp_index += 1
        # self.cell_pool.append(cell)
        # self.cell_pool.update(observation=np.zeros(128), trajectory=None, score=-np.infty, state=None)
        self.env.set_param_values([self.cell_pool.pool_filename], db_filename=True, debug=False)
        self.env.set_param_values([self.cell_pool.key_list], key_list=True, debug=False)
        self.env.set_param_values([self.cell_pool.max_value], max_value=True, debug=False)
        self.env.set_param_values([None], robustify_state=True, debug=False)

        # for cell in d_pool.values():

        # if cell.score == 0.0 and cell.reward >= self.max_cum_reward and cell.observation is not None and cell.parent is not None:
        # pdb.set_trace()
        # print(cell.observation, cell.score, cell.reward)

        # self.max_cum_reward = cell.reward
        # self.best_cell = cell
        # pdb.set_trace()

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

    def get_itr_snapshot(self, itr):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        # pdb.set_trace()
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
        )

    def optimize_policy(self, itr, samples_data):

        start = time.time()

        # pool_DB = db.DB()
        # pool_DB.open(self.db_filename, dbname=None, dbtype=db.DB_HASH, flags=db.DB_CREATE)
        # d_pool = shelve.Shelf(pool_DB, protocol=pickle.HIGHEST_PROTOCOL)
        d_pool = self.cell_pool.open_pool()

        new_cells = 0
        total_cells = 0
        # self.cell_pool.d_pool.open()
        # pdb.set_trace()
        for i in range(samples_data['observations'].shape[0]):
            sys.stdout.write("\rProcessing Trajectory {0} / {1}".format(i, samples_data['observations'].shape[0]))
            sys.stdout.flush()
            cum_reward = 0
            cum_traj = np.array([])
            observation = None
            is_terminal = False
            is_goal = False
            root_step = 0
            for j in range(samples_data['observations'].shape[1]):
                # pdb.set_trace()
                # If action only (black box) we search based on history of actions
                if self.env.blackbox_sim_state:
                    # pdb.set_trace()
                    observation_data = samples_data['env_infos']['actions']
                    action_data = samples_data['env_infos']['actions']
                    # observation = samples_data['actions'][i, j, :]
                else:
                    # else (white box) use simulation state
                    observation_data = samples_data['observations']
                    action_data = samples_data['env_infos']['actions']
                    # observation = samples_data['observations'][i, j, :]
                # chosen = 0
                if j == 0:
                    # chosen = 1
                    try:
                        # root_cell = d_pool[str(hash(observation_data[i, j, :].tostring()))]
                        # Get the chosen cell that was root of this rollout
                        # root_obs = self.downsample(obs=samples_data['env_infos']['root_action'][i,j,:],
                        #                            step=samples_data['env_infos']['state'][i, j, -1]-1)
                        root_cell = d_pool[str(hash(samples_data['env_infos']['root_action'][i, j, :].tostring()))]
                        # Update the chosen/visited count
                        self.cell_pool.d_update(d_pool=d_pool,
                                                observation=root_cell.observation,
                                                action=root_cell.action,
                                                trajectory=root_cell.trajectory,
                                                score=root_cell.score,
                                                state=root_cell.state,
                                                parent=root_cell.parent,
                                                reward=root_cell.reward,
                                                is_goal=root_cell.is_goal,
                                                is_terminal=root_cell.is_terminal,
                                                chosen=1)
                        # self.cell_pool.d_update(d_pool=d_pool,observation=root_cell.observation,trajectory=root_cell.trajectory,score=root_cell.score,state=root_cell.state,parent=root_cell.parent,reward=root_cell.reward,chosen=1)
                        # Update trajectory info to root cell state
                        cum_reward = root_cell.reward
                        cum_traj = root_cell.trajectory
                        if cum_traj.shape[0] > 0:
                            cum_traj = np.concatenate((cum_traj, root_cell.action.reshape((1, 6))), axis=0)
                        parent = str(hash(samples_data['env_infos']['root_action'][i, j, :].tostring()))
                        root_step = root_cell.state[-1] + 1
                        # pdb.set_trace()
                    except BaseException:
                        print('----------ERROR - failed to retrieve root cell--------------------')
                        pdb.set_trace()
                        break
                else:
                    parent = str(hash(observation.tostring()))
                    # if cum_reward == 0 or cum_reward  <-1e8:
                    #     pdb.set_trace()

                if np.all(observation_data[i, j, :] == 0):
                    continue
                observation = self.downsample(observation_data[i, j, :], root_step + j)
                action = action_data[i, j, :]
                # trajectory = observation_data[i, 0:j, :]
                trajectory = action_data[i, 0:j, :]
                if cum_traj.shape[0] > 0:
                    trajectory = np.concatenate((cum_traj, trajectory), axis=0)
                score = samples_data['rewards'][i, j]
                cum_reward += score
                state = samples_data['env_infos']['state'][i, j, :]
                is_terminal = samples_data['env_infos']['is_terminal'][i, j]
                is_goal = samples_data['env_infos']['is_goal'][i, j]
                # if j >48:
                #     print(j)
                #     pdb.set_trace()
                if self.cell_pool.d_update(d_pool=d_pool,
                                           observation=observation,
                                           action=action,
                                           trajectory=trajectory,
                                           score=score,
                                           state=state,
                                           parent=parent,
                                           is_goal=is_goal,
                                           is_terminal=is_terminal,
                                           reward=cum_reward,
                                           chosen=0):
                    new_cells += 1
                total_cells += 1
                # pdb.set_trace()
            if cum_reward > self.max_cum_reward and observation is not None:
                self.max_cum_reward = cum_reward
                self.best_cell = d_pool[str(hash(observation.tostring()))]
            # if cum_reward > -100:
            #     pdb.set_trace()
            # pdb.set_trace()
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(new_cells, " new cells (", 100 * new_cells / total_cells, "%)")
        print(total_cells, " samples processed in ", time.time() - start, " seconds")

        self.cell_pool.sync_and_close_pool(cell_pool_shelf=d_pool)
        # self.cell_pool.d_pool.close()
        # self.env.set_param_values([self.cell_pool], pool=True, debug=True)
        self.env.set_param_values([self.cell_pool.key_list], key_list=True, debug=False)
        self.env.set_param_values([self.cell_pool.max_value], max_value=True, debug=False)

        if self.save_paths_gap != 0 and self.save_paths_path is not None and itr % self.save_paths_gap == 0:
            with open(self.save_paths_path + '/paths_itr_' + str(itr) + '.p', 'wb') as f:
                pickle.dump(samples_data, f)

        if os.path.getsize(self.cell_pool.pool_filename) / 1000 / 1000 / 1000 > self.max_db_size:
            print('------------ERROR: MAX DB SIZE REACHED------------')
            sys.exit()
        print('\n---------- Max Score: ', self.max_cum_reward, ' ----------------\n')
        tabular.record('MaxReturn', self.max_cum_reward)

    def downsample(self, obs, step=None):
        # import pdb; pdb.set_trace()
        obs = obs * 1000
        # if step is None:
        #     step = self._step

        return np.concatenate((np.array([step]), obs), axis=0).astype(int)
