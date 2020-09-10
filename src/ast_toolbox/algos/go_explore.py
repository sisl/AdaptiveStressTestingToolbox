"""Implementation of the `Go-Explore <https://arxiv.org/abs/1901.10995>`_ algorithm."""
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
    r"""A representation of a state visited during exploration.

    Parameters
    ----------
    use_score_weight : bool
        Whether or not to scale the cell's fitness by a function of the cell's score
    """

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
        r"""Removes cached properties so they will be recalculated on next access.

        Parameters
        ----------
        cached_property : str
            The cached_property key to remove from the class dict.
        """
        if cached_property in self.__dict__:
            del self.__dict__[cached_property]

    @property
    def is_root(self):
        r"""Checks if the cell is the root of the tree (trajectory length is 0).

        Returns
        -------
        bool
            Whether the cell is root or not
        """
        return len(self.trajectory) == 0

    @property
    def step(self):
        r"""How many steps led to the current cell.

        Returns
        -------
        int
            Length of the trajectory.
        """
        return len(self.trajectory)

    @property
    def reward(self):
        r"""The reward obtained in the current cell.

        Returns
        -------
        float
            The reward.
        """
        return self._reward

    @reward.setter
    def reward(self, value):
        self._reward = value
        self.reset_cached_property('score_weight')
        self.reset_cached_property('fitness')

    @property
    def value_approx(self):
        r"""The approximate value of the current cell, based on backpropigation of previous rollouts.

        Returns
        -------
        float
            The value approximation.
        """
        return self._value_approx

    @value_approx.setter
    def value_approx(self, value):
        self._value_approx = value
        self.reset_cached_property('score_weight')
        self.reset_cached_property('fitness')

    @property
    def is_terminal(self):
        r"""Whether or not the current cell is a terminal state.

        Returns
        -------
        bool
            Is the current cell terminal.
        """
        return self._is_terminal

    @is_terminal.setter
    def is_terminal(self, value):
        self._is_terminal = value
        self.reset_cached_property('score_weight')
        self.reset_cached_property('fitness')

    @property
    def is_goal(self):
        r"""Whether or not the current cell is a goal state.

        Returns
        -------
        bool
            Is the current cell a goal.
        """
        return self._is_goal

    @is_goal.setter
    def is_goal(self, value):
        self._is_goal = value
        self.reset_cached_property('score_weight')
        self.reset_cached_property('fitness')

    @property
    def score(self):
        r"""The `score` obtained in the current cell.

        Returns
        -------
        float
            The score.
        """
        return self._score

    @score.setter
    def score(self, value):
        self._score = value
        self.reset_cached_property('score_weight')
        self.reset_cached_property('fitness')

    @property
    def times_visited(self):
        r"""How many times the current cell has been visited during all rollouts.

        Returns
        -------
        int
            Number of times visited.
        """
        return self._times_visited

    @times_visited.setter
    def times_visited(self, value):
        self._times_visited = value
        self.reset_cached_property('times_visited_subscore')
        self.reset_cached_property('count_subscores')
        self.reset_cached_property('fitness')

    @property
    def times_chosen(self):
        r"""How many times the current cell has been chosen to start a rollout.

        Returns
        -------
        int
            Number of times chosen.
        """
        return self._times_chosen

    @times_chosen.setter
    def times_chosen(self, value):
        self._times_chosen = value
        self.reset_cached_property('times_chosen_subscore')
        self.reset_cached_property('count_subscores')
        self.reset_cached_property('fitness')

    @property
    def times_chosen_since_improved(self):
        r"""How many times the current cell has been chosen to start a rollout since the last time the cell was updated
        with an improved score or trajectory.

        Returns
        -------
        int
            Number of times chosen since last improved.
        """
        return self._times_chosen_since_improved

    @times_chosen_since_improved.setter
    def times_chosen_since_improved(self, value):
        self._times_chosen_since_improved = value
        self.reset_cached_property('times_chosen_since_improved')
        self.reset_cached_property('count_subscores')
        self.reset_cached_property('fitness')

    @cached_property
    def fitness(self):
        r"""The `fitness` score of the cell. Cells are sampled with probability proportional to their `fitness` score.

        Returns
        -------
        float
            The fitness score of the cell.
        """
        # return max(1, self.score)
        return self.score_weight * (self.count_subscores + 1)

    @cached_property
    def count_subscores(self):
        r"""A function of `times_chosen_subscore`, `times_chosen_since_improved_subscore`, and `times_visited_subscore`
        that is used in calculating the cell's `fitness` score.

        Returns
        -------
        float
            The count subscore of the cell.
        """
        return (self.times_chosen_subscore +
                self.times_chosen_since_improved_subscore +
                self.times_visited_subscore)

    @cached_property
    def times_chosen_subscore(self):
        r"""A function of `times_chosen` that is used in calculating the cell's `times_chosen_subscore` score.

        Returns
        -------
        float
            The `times_chosen_subscore`
        """
        weight = 0.1
        power = 0.5
        eps1 = 0.001
        eps2 = 0.00001
        return weight * (1 / (self.times_chosen + eps1)) ** power + eps2

    @cached_property
    def times_chosen_since_improved_subscore(self):
        r"""A function of `times_chosen_since_improved` that is used in calculating the cell's
        `times_chosen_since_improved_subscore` score.

        Returns
        -------
        float
            The `times_chosen_since_improved_subscore`
        """
        weight = 0.0
        power = 0.5
        eps1 = 0.001
        eps2 = 0.00001
        return weight * (1 / (self.times_chosen_since_improved + eps1)) ** power + eps2

    @cached_property
    def times_visited_subscore(self):
        r"""A function of `_times_visited` that is used in calculating the cell's
        `times_visited_subscore` score.

        Returns
        -------
        float
            The `times_visited_subscore`
        """
        weight = 0.3
        power = 0.5
        eps1 = 0.001
        eps2 = 0.00001
        return weight * (1 / (self._times_visited + eps1)) ** power + eps2

    @cached_property
    def score_weight(self):
        r"""A heuristic function basedon the cell's score, and other values, to bias the rollouts towards high-scoring
        areas.

        Returns
        -------
        float
            The cell's `score_weight`
        """
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
    r"""A hashtree data structure containing and updating all of the cells seen during rollouts.

    Parameters
    ----------
    filename : str, optional
        The base name for the database files. The CellPool saves a `[filename]_pool.dat` and a `[filename]_meta.dat`.
    discount : float, optional
        Discount factor used in calculating a cell's value approximation.
    use_score_weight : bool
        Whether or not to scale a cell's fitness by a function of the cell's score
    """

    def __init__(self, filename='database', discount=0.99, use_score_weight=True):

        self.length = 0
        self._filename = filename
        self.discount = discount

        self.key_list = []
        self.goal_dict = {}
        self.terminal_dict = {}
        self.max_value = 0
        self.max_score = -np.inf
        self.max_reward = -np.inf
        self.best_cell = None

        self.use_score_weight = use_score_weight

    def save(self):
        r"""Save the CellPool to disk.

        """
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
        r"""Load a CellPool from disk.

        Parameters
        ----------
        cell_pool_shelf : `shelve.Shelf <https://docs.python.org/3/library/shelve.html#shelve.Shelf>`_
            A `shelve.Shelf` wrapping a bsddb3 database.
        """
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
        r"""Open the database that the CellPool uses to store cells.

        Parameters
        ----------
        dbname : string

        dbtype : int, optional
            Specifies the type of database to open. Use enumerations provided by
            `bsddb3 <https://www.jcea.es/programacion/pybsddb_doc/db.html#open>`_.
        flags : int, optional
            Specifies the configuration of the database to open. Use enumerations provided by
            `bsddb3 <https://www.jcea.es/programacion/pybsddb_doc/db.html#open>`_.
        protocol : int, optional
            Specifies the data stream format used by
            `pickle <https://docs.python.org/3/library/pickle.html#data-stream-format>`_.
        overwrite : bool, optional
            Indicates if an existing database should be overwritten if found.
        Returns
        -------
        cell_pool_shelf : `shelve.Shelf <https://docs.python.org/3/library/shelve.html#shelve.Shelf>`_
            A `shelve.Shelf` wrapping a bsddb3 database.
        """
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
        r"""Syncs the pool, ensuring that the database on disk is up-to-date.

        Parameters
        ----------
        cell_pool_shelf : `shelve.Shelf <https://docs.python.org/3/library/shelve.html#shelve.Shelf>`_
            A `shelve.Shelf` wrapping a bsddb3 database.
        """
        # We can't save our database as a class attribute due to pickling errors.
        # To prevent errors from code repeat, this convenience function syncs the given database and
        # saves the latest meta data.
        cell_pool_shelf.sync()
        self.save()

    def close_pool(self, cell_pool_shelf):
        r"""Close the database that the CellPool uses to store cells.

        Parameters
        ----------
        cell_pool_shelf : `shelve.Shelf <https://docs.python.org/3/library/shelve.html#shelve.Shelf>`_
            A `shelve.Shelf` wrapping a bsddb3 database.
        """
        # We can't save our database as a class attribute due to pickling errors.
        # To prevent errors from code repeat, this convenience function closes the given database and
        # saves the latest meta data.
        cell_pool_shelf.close()
        self.save()

    def sync_and_close_pool(self, cell_pool_shelf):
        r"""Sync and then close the database that the CellPool uses to store cells.

        Parameters
        ----------
        cell_pool_shelf : `shelve.Shelf <https://docs.python.org/3/library/shelve.html#shelve.Shelf>`_
            A `shelve.Shelf` wrapping a bsddb3 database
        """
        # We can't save our database as a class attribute due to pickling errors.
        # To prevent errors from code repeat, this convenience function syncs and closes the given
        # database and saves the latest meta data.
        cell_pool_shelf.sync()
        cell_pool_shelf.close()
        self.save()

    def delete_pool(self):
        r"""Remove the CellPool files saved on disk.

        """
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.pool_filename)
            os.remove(self.meta_filename)

    @cached_property
    def pool_filename(self):
        r"""The CellPool database filename.

        Returns
        -------
        str
            The CellPool database filename.
        """
        return self._filename + '_pool.dat'

    @cached_property
    def meta_filename(self):
        r"""The CellPool metadata filename.

        Returns
        -------
        str
            The CellPool metadata filename.
        """
        return self._filename + '_meta.dat'

    def d_update(self, cell_pool_shelf, observation, action, trajectory, score, state,
                 parent=None, is_terminal=False, is_goal=False, reward=-np.inf, chosen=0):
        r"""Runs the update algorithm for the CellPool. The process is:
        1. Create a cell from the given data.
        2. Check if the cell already exists in the CellPool.
        3. If the cell already exists and our version is better (higher fitness or shorter trajectory), update the
        existing cell.
        4. If the cell already exists and our version is not better, end.
        5. If the cell does not already exists, add the new cell to the CellPool

        Parameters
        ----------
        cell_pool_shelf : `shelve.Shelf <https://docs.python.org/3/library/shelve.html#shelve.Shelf>`_
            A `shelve.Shelf` wrapping a bsddb3 database.
        observation : array_like
            The observation seen in the current cell.
        action : array_like
            The action taken in the current cell.
        trajectory : array_like
            The trajectory leading to the current cell.
        score : float
            The score at the current cell.
        state : array_like
            The cloned simulation state at the current cell, used for resetting if chosen to start a rollout.
        parent : int, optional
            The hash key of the cell immediately preceding the current cell in the trajectory.
        is_terminal : bool, optional
            Whether the current cell is a terminal state.
        is_goal : bool, optional
            Whether the current cell is a goal state.
        reward : float, optional
            The reward obtained at the current cell.
        chosen : int, optional
            Whether the current cell was chosen to start the rollout.
        Returns
        -------
        bool
            True if a new cell was added to the CellPool, False otherwise
        """
        # pdb.set_trace()
        # This tests to see if the observation is already in the matrix

        obs_hash = str(hash(observation.tostring()))
        if obs_hash not in cell_pool_shelf:
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

            cell_pool_shelf[obs_hash] = cell
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

            self.value_approx_update(value=cell.value_approx, obs_hash=cell.parent, cell_pool_shelf=cell_pool_shelf)

            return True
        else:
            cell = cell_pool_shelf[obs_hash]
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
            cell_pool_shelf[obs_hash] = cell
            if cell.fitness > self.max_value:
                self.max_value = cell.fitness
            if cell.score > self.max_score:
                self.max_score = score

            self.value_approx_update(value=cell.value_approx, obs_hash=cell.parent, cell_pool_shelf=cell_pool_shelf)

        return False

    def value_approx_update(self, value, obs_hash, cell_pool_shelf):
        r"""Recursively calculate a value approximation through back-propagation.

        Parameters
        ----------
        value : Value approximation of the previous cell.
        obs_hash : Hash key of the current cell.
        cell_pool_shelf : `shelve.Shelf <https://docs.python.org/3/library/shelve.html#shelve.Shelf>`_
            A `shelve.Shelf` wrapping a bsddb3 database.
        """
        if obs_hash is not None:
            cell = cell_pool_shelf[obs_hash]
            v = cell.score + self.discount * value
            cell.value_approx = (v - cell.value_approx) / cell.times_visited + cell.value_approx
            cell_pool_shelf[obs_hash] = cell
            if cell.parent is not None:
                self.value_approx_update(value=cell.value_approx, obs_hash=cell.parent, cell_pool_shelf=cell_pool_shelf)


class GoExplore(BatchPolopt):
    r"""Implementation of the Go-Explore[1]_ algorithm that is compatible with AST[2]_.
    Parameters
    ----------
    db_filename : str
        The base path and name for the database files. The CellPool saves a `[filename]_pool.dat` and a `[filename]_meta.dat`.
    max_db_size : int
        Maximum allowable size (in GB) of the CellPool database.
        Algorithm will immediately stop and exit if this size is exceeded.
    env : :py:class:`ast_toolbox.envs.go_explore_ast_env.GoExploreASTEnv`
        The environment.
    env_spec : :py:class:`garage.envs.EnvSpec`
        Environment specification.
    policy : :py:class:`garage.tf.policies.Policy`
        The policy.
    baseline : :py:class:`garage.np.baselines.Baseline`
        The baseline.
    save_paths_gap : int, optional
        How many epochs to skip between saving out full paths. Set to `1` to save every epoch.
        Set to `0` to disable saving.
    save_paths_path : str, optional
        Path to the directory where paths should be saved. Set to `None` to disable saving.
    overwrite_db : bool, optional
        Indicates if an existing database should be overwritten if found.
    use_score_weight : bool
        Whether or not to scale the cell's fitness by a function of the cell's score
    kwargs :
        Keyword arguments passed to :doc:`garage.tf.algos.BatchPolopt <garage:_apidoc/garage.tf.algos.batch_polopt>`

    References
    ----------
    .. [1] Ecoffet, Adrien, et al. "Go-explore: a new approach for hard-exploration problems."
        arXiv preprint arXiv:1901.10995 (2019). `<https://arxiv.org/abs/1901.10995>`_
    .. [2] Koren, Mark, and Mykel J. Kochenderfer. "Adaptive Stress Testing without Domain Heuristics
        using Go-Explore." arXiv preprint arXiv:2004.04292 (2020). `<https://arxiv.org/abs/2004.04292>`_
    """

    def __init__(self,
                 db_filename,
                 max_db_size,
                 env,
                 env_spec,
                 policy,
                 baseline,
                 save_paths_gap=0,
                 save_paths_path=None,
                 overwrite_db=True,
                 use_score_weight=True,
                 **kwargs):

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
        """Obtain samplers and start actual training for each epoch.

        Parameters
        ----------
        runner : :py:class:`garage.experiment.LocalRunner <garage:garage.experiment.LocalRunner>`
            ``LocalRunner`` is passed to give algorithm the access to ``runner.step_epochs()``, which provides services
            such as snapshotting and sampler control.

        Returns
        -------
        last_return : :py:class:`ast_toolbox.algos.go_explore.Cell`
            The highest scoring cell found so far
        """
        last_return = None
        self.policy = self.go_explore_policy
        for epoch in runner.step_epochs():
            runner.step_path = runner.obtain_samples(runner.step_itr)
            last_return = self.train_once(runner.step_itr, runner.step_path)
            runner.step_itr += 1

        return last_return

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Parameters
        ----------
        itr : int
            Iteration number.
        paths : list[dict]
            A list of collected paths.

        Returns
        -------
        best_cell : :py:class:`ast_toolbox.algos.go_explore.Cell`
            The highest scoring cell found so far
        """
        paths = self.process_samples(itr, paths)

        self.log_diagnostics(paths)
        logger.log('Optimizing policy...')
        self.optimize_policy(itr, paths)
        return self.best_cell

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        # if self.robustify:
        #     self.env.set_param_values([None], robustify_state=True, debug=False)
        self.max_cum_reward = -np.inf

        self.cell_pool = CellPool(filename=self.db_filename, use_score_weight=self.use_score_weight)

        d_pool = self.cell_pool.open_pool(overwrite=self.overwrite_db)
        if len(self.cell_pool.key_list) == 0:
            obs, state = self.env.get_first_cell()
            self.cell_pool.d_update(cell_pool_shelf=d_pool, observation=self.downsample(obs, step=-1), action=obs,
                                    trajectory=np.array([]), score=0.0, state=state, reward=0.0, chosen=0)
            self.cell_pool.sync_pool(cell_pool_shelf=d_pool)

        self.max_cum_reward = self.cell_pool.max_reward
        self.best_cell = self.cell_pool.best_cell

        self.cell_pool.close_pool(cell_pool_shelf=d_pool)
        self.env.set_param_values([self.cell_pool.pool_filename], db_filename=True, debug=False)
        self.env.set_param_values([self.cell_pool.key_list], key_list=True, debug=False)
        self.env.set_param_values([self.cell_pool.max_value], max_value=True, debug=False)
        self.env.set_param_values([None], robustify_state=True, debug=False)

    def get_itr_snapshot(self, itr):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.

        Parameters
        ----------
        itr : int
            The current epoch number.

        Returns
        -------
        dict
            A dict containing the current iteration number, the current policy, and the current baseline.
        """
        # pdb.set_trace()
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
        )

    def optimize_policy(self, itr, samples_data):
        """Optimize the policy using the samples.

        Parameters
        ----------
        itr : int
            The current epoch number.
        samples_data : dict
            The data from the sampled rollouts.
        """
        start = time.time()

        d_pool = self.cell_pool.open_pool()

        new_cells = 0
        total_cells = 0

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
                        root_cell = d_pool[str(hash(samples_data['env_infos']['root_action'][i, j, :].tostring()))]
                        # Update the chosen/visited count
                        self.cell_pool.d_update(cell_pool_shelf=d_pool,
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
                if self.cell_pool.d_update(cell_pool_shelf=d_pool,
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
        # import pdb; pdb.set_trace()
        obs = obs * 1000

        return np.concatenate((np.array([step]), obs), axis=0).astype(int)
