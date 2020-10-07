import gym.utils.seeding as seeding
import numpy as np
from dowel import logger
from dowel import tabular
from garage.misc import tensor_utils as np_tensor_utils
from garage.tf.algos.batch_polopt import BatchPolopt
from garage.tf.misc import tensor_utils


class GA(BatchPolopt):
    """Deep Genetic Algorithm from Such et al.  [1]_.

    Parameters
    ----------
    top_paths : :py:class:`ast_toolbox.mcts.BoundedPriorityQueues`, optional
        The bounded priority queue to store top-rewarded trajectories.
    step_size : float, optional
        Standard deviation for each mutation.
    step_size_anneal : float, optional
        The linear annealing rate of step_size after each iteration.
    pop_size : int, optional
        The population size
    truncation_size : int, optional
        The number of top-performed individuals that are chosen as parents.
    keep_best : int, optional
        The number of top-performed individuals that remain unchanged for next generation.
    f_F : string, optional
        The method used to calculate fitness: 'mean' for the average return, 'max' for the max return.
    log_interval : int, optional
        The log interval in terms of environment calls.
    kwargs :
        Keyword arguments passed to :doc:`garage.tf.algos.BatchPolopt <garage:_apidoc/garage.tf.algos>`.

    References
    ----------
    .. [1] Such, Felipe Petroski, et al. "Deep neuroevolution: Genetic algorithms are a competitive
    alternative for training deep neural networks for reinforcement learning."
     arXiv:1712.06567 (2017).
    """

    def __init__(
            self,
            top_paths=None,
            n_itr=2,
            batch_size=500,
            step_size=0.01,
            step_size_anneal=1.0,
            pop_size=5,
            truncation_size=2,
            keep_best=1,
            f_F="mean",
            log_interval=4000,
            init_step=1.0,
            **kwargs):

        self.top_paths = top_paths
        self.best_mean = -np.inf
        self.best_var = 0.0
        self.n_itr = n_itr
        self.batch_size = batch_size
        self.step_size = step_size
        self.step_size_anneal = step_size_anneal
        self.pop_size = pop_size
        self.truncation_size = truncation_size
        self.keep_best = keep_best
        self.f_F = f_F
        self.log_interval = log_interval
        self.init_step = init_step

        self.seeds = np.zeros([n_itr, pop_size], dtype=int)
        self.magnitudes = np.zeros([n_itr, pop_size])
        self.parents = np.zeros(pop_size, dtype=int)
        self.np_random, seed = seeding.np_random()  # used in set_params
        super(GA, self).__init__(**kwargs)

    def initial(self):
        """Initiate trainer internal parameters.
        """
        self.seeds[0, :] = np.random.randint(low=0, high=int(2**16),
                                             size=(1, self.pop_size))
        self.magnitudes[0, :] = self.init_step * np.ones(self.pop_size)
        self.policy.set_param_values(self.policy.get_param_values())
        self.stepNum = 0

    def init_opt(self):
        """Initiate trainer internal tensorflow operations.
        """
        return dict()

    def train(self, runner):
        """Start training.

        Parameters
        ----------
        runner : :py:class:`garage.experiment.LocalRunner <garage:garage.experiment.LocalRunner>`
            ``LocalRunner`` is passed to give algorithm the access to ``runner.step_epochs()``, which provides services
            such as snapshotting and sampler control.
        """
        self.initial()

        for itr in runner.step_epochs():
            all_paths = {}
            for p in range(self.pop_size):
                with logger.prefix('idv #%d | ' % p):
                    logger.log("Updating Params")
                    self.set_params(itr, p)
                    logger.log("Obtaining samples...")
                    paths = self.obtain_samples(itr, runner)
                    logger.log("Processing samples...")
                    samples_data = self.process_samples(itr, paths)

                    # all_paths[p]=paths
                    all_paths[p] = samples_data

                    # logger.log("Logging diagnostics...")
                    # self.log_diagnostics(paths)

            logger.log("Optimizing Population...")
            self.optimize_policy(itr, all_paths)
            self.step_size = self.step_size * self.step_size_anneal
            self.record_tabular(itr)
            runner.step_itr += 1
        return None

    def record_tabular(self, itr):
        """Record training performace per-iteration.

        Parameters
        ----------
        itr : int
            The iteration number.
        """
        tabular.record('Itr', itr)
        tabular.record('StepNum', self.stepNum)
        # This causes tabular logging inconsistant
        # if self.top_paths is not None:
        #     for (topi, path) in enumerate(self.top_paths):
        #         tabular.record('reward ' + str(topi), path[0])
        tabular.record('BestMean', self.best_mean)
        tabular.record('BestVar', self.best_var)
        tabular.record('StepSize', self.step_size)
        tabular.record('Max Magnitude', np.max(self.magnitudes[itr, :]))
        tabular.record('Min Magnitude', np.min(self.magnitudes[itr, :]))
        tabular.record('Mean Magnitude', np.mean(self.magnitudes[itr, :]))
        self.extra_recording(itr)

    def extra_recording(self, itr):
        """Record extra training statistics per-iteration.

        Parameters
        ----------
        itr : int
            The iteration number.
        """
        return None

    def set_params(self, itr, p):
        """Set the current policy paramter to the specified iteration and individual.

        Parameters
        ----------
        itr : int
            The iteration number.
        p : int
            The individual index.
        """
        for i in range(itr + 1):
            # print("seed: ", self.seeds[i,p])
            self.np_random.seed(int(self.seeds[i, p]))
            if i == 0:  # first generation
                param_values = self.policy.get_param_values(trainable=True)
                param_values = self.magnitudes[i, p] * self.np_random.normal(size=param_values.shape)

                # param_values = init_policy_np(self.policy, self.np_random)

                # params = self.policy.get_params()
                # sess = tf.get_default_session()
                # sess.run(tf.variables_initializer(params))
                # param_values = self.policy.get_param_values()
            elif self.seeds[i, p] != 0:
                param_values = param_values + self.magnitudes[i, p] * self.np_random.normal(size=param_values.shape)
        self.policy.set_param_values(param_values, trainable=True)

    def get_fitness(self, itr, all_paths):
        """Calculate the fitness of the collexted paths.

        Parameters
        ----------
        itr : int
            The iteration number.
        all_paths : list[dict]
            The collected paths from the sampler.

        Returns
        ----------
        fitness : list[float]
            The list of fitness of each individual.
        """
        fitness = np.zeros(self.pop_size)
        for p in range(self.pop_size):
            rewards = all_paths[p]["rewards"]
            valid_rewards = rewards * all_paths[p]["valids"]
            path_rewards = np.sum(valid_rewards, -1)
            if self.f_F == "max":
                fitness[p] = np.max(path_rewards)
            else:
                fitness[p] = np.mean(path_rewards)
        return fitness

    def select_parents(self, fitness):
        """Select the individuals to be the parents of the next generation.

        Parameters
        ----------
        fitness : list[float]
            The list of fitness of each individual.
        """
        sort_indx = np.flip(np.argsort(fitness), axis=0)
        self.parents[0:self.truncation_size] = sort_indx[0:self.truncation_size]
        self.parents[self.truncation_size:self.pop_size] = \
            sort_indx[np.random.randint(low=0, high=self.truncation_size, size=self.pop_size - self.truncation_size)]

    def mutation(self, itr, new_seeds, new_magnitudes, all_paths):
        """Generate new random seeds and magnitudes for the next generation.

        The first self.keep_best seeds are set to no-mutation value (0).

        Parameters
        ----------
        itr : int
            The iteration number.
        new_seeds : :py:class:`numpy.ndarry`
            The original seeds.
        new_magnitudes : :py:class:`numpy.ndarry`
            The original magnitudes.
        all_paths : list[dict]
            The collected paths from the sampler.

        Returns
        -------
        new_seeds : :py:class:`numpy.ndarry`
            The new seeds.
        new_magnitudes : :py:class:`numpy.ndarry`
            The new magnitudes.
        """
        if itr + 1 < self.n_itr:
            new_seeds[itr + 1, :] = np.random.randint(low=0, high=int(2**32),
                                                      size=(1, self.pop_size))
            new_magnitudes[itr + 1, :] = self.step_size
            for i in range(0, self.keep_best):
                new_seeds[itr + 1, i] = 0
        return new_seeds, new_magnitudes

    def optimize_policy(self, itr, all_paths):
        """Update the population represented by self.seeds and self.parents.

        Parameters
        ----------
        itr : int
            The iteration number.
        all_paths : list[dict]
            The collected paths from the sampler.
        """
        fitness = self.get_fitness(itr, all_paths)
        self.select_parents(fitness)
        new_seeds = np.zeros_like(self.seeds)
        new_seeds[:, :] = self.seeds[:, self.parents]
        new_magnitudes = np.zeros_like(self.magnitudes)
        new_magnitudes[:, :] = self.magnitudes[:, self.parents]
        if itr + 1 < self.n_itr:
            new_seeds, new_magnitudes = self.mutation(itr, new_seeds, new_magnitudes, all_paths)
        self.seeds = new_seeds
        self.magnitudes = new_magnitudes
        return dict()

    def obtain_samples(self, itr, runner):
        """Collect rollout samples using the current policy paramter.

        Parameters
        ----------
        itr : int
            The iteration number.
        runner : :py:class:`garage.experiment.LocalRunner <garage:garage.experiment.LocalRunner>`
            ``LocalRunner`` is passed to give algorithm the access to ``runner.obtain_samples()``,
            which collects rollout paths from the sampler.

        Returns
        -------
        paths : list[dict]
            The collected paths from the sampler.
        """
        self.stepNum += self.batch_size
        # paths = self.sampler.obtain_samples(itr)
        paths = runner.obtain_samples(runner.step_itr)
        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        if np.mean(undiscounted_returns) > self.best_mean:
            self.best_mean = np.mean(undiscounted_returns)
            self.best_var = np.var(undiscounted_returns)
        if not (self.top_paths is None):
            action_seqs = [path["actions"] for path in paths]
            [self.top_paths.enqueue(
                action_seq, R, make_copy=True) for (action_seq, R) in zip(action_seqs, undiscounted_returns)]
        return paths

    def process_samples(self, itr, paths):
        """Return processed sample data based on the collected paths.

        Parameters
        ----------
        itr : int
            The iteration number.
        paths : list[dict]
            The collected paths from the sampler.

        Returns
        -------
        samples_data : dict
            Processed sample data with same trajectory length (padded with 0)
        """
        baselines = []
        returns = []

        max_path_length = self.max_path_length

        if self.flatten_input:
            paths = [
                dict(
                    observations=(self.env_spec.observation_space.flatten_n(
                        path['observations'])),
                    actions=(
                        self.env_spec.action_space.flatten_n(  # noqa: E126
                            path['actions'])),
                    rewards=path['rewards'],
                    env_infos=path['env_infos'],
                    agent_infos=path['agent_infos']) for path in paths
            ]
        else:
            paths = [
                dict(
                    observations=path['observations'],
                    actions=(
                        self.env_spec.action_space.flatten_n(  # noqa: E126
                            path['actions'])),
                    rewards=path['rewards'],
                    env_infos=path['env_infos'],
                    agent_infos=path['agent_infos']) for path in paths
            ]

        if hasattr(self.baseline, 'predict_n'):
            all_path_baselines = self.baseline.predict_n(paths)
        else:
            all_path_baselines = [
                self.baseline.predict(path) for path in paths
            ]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = (path['rewards'] + self.discount * path_baselines[1:] -
                      path_baselines[:-1])
            path['advantages'] = np_tensor_utils.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path['deltas'] = deltas

        for idx, path in enumerate(paths):
            # baselines
            path['baselines'] = all_path_baselines[idx]
            baselines.append(path['baselines'])

            # returns
            path['returns'] = np_tensor_utils.discount_cumsum(
                path['rewards'], self.discount)
            returns.append(path['returns'])

        # make all paths the same length
        obs = [path['observations'] for path in paths]
        obs = tensor_utils.pad_tensor_n(obs, max_path_length)

        actions = [path['actions'] for path in paths]
        actions = tensor_utils.pad_tensor_n(actions, max_path_length)

        rewards = [path['rewards'] for path in paths]
        rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

        returns = [path['returns'] for path in paths]
        returns = tensor_utils.pad_tensor_n(returns, max_path_length)

        baselines = tensor_utils.pad_tensor_n(baselines, max_path_length)

        agent_infos = [path['agent_infos'] for path in paths]
        agent_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length)
            for p in agent_infos
        ])

        env_infos = [path['env_infos'] for path in paths]
        env_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos
        ])

        valids = [np.ones_like(path['returns']) for path in paths]
        valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        # average_discounted_return = (np.mean(
        #     [path['returns'][0] for path in paths]))

        undiscounted_returns = [sum(path['rewards']) for path in paths]
        self.episode_reward_mean.extend(undiscounted_returns)

        # ent = np.sum(self.policy.distribution.entropy(agent_infos) *
        #              valids) / np.sum(valids)

        samples_data = dict(
            observations=obs,
            actions=actions,
            rewards=rewards,
            baselines=baselines,
            returns=returns,
            valids=valids,
            agent_infos=agent_infos,
            env_infos=env_infos,
            paths=paths,
            average_return=np.mean(undiscounted_returns),
        )

        return samples_data

    def get_itr_snapshot(self, itr, samples_data):
        """Get the snapshot of the current population.

        Parameters
        ----------
        itr : int
            The iteration number.
        samples_data : dict
            The processed data samples.

        Returns
        -------
        snaposhot : dict
            The training snapshot.
        """
        return dict(
            itr=itr,
            policy=self.policy,
            seeds=self.seeds,
            magnitudes=self.magnitudes,
        )
