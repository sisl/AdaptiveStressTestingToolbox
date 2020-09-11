"""Module for parallel sampling a batch of rollouts"""
import numpy as np
import tensorflow as tf
from garage.sampler.base import BaseSampler
from garage.sampler.stateful_pool import singleton_pool
from garage.sampler.utils import truncate_paths

from ast_toolbox.rewards import ExampleAVReward
from ast_toolbox.samplers import parallel_sampler
from ast_toolbox.simulators import ExampleAVSimulator


def worker_init_tf(g):
    """Initialize the tf.Session on a worker.

    Parameters
    ----------
    g : :py:class:`garage.sampler.stateful_pool.SharedGlobal`
        SharedGlobal class from :py:mod:`garage.sampler.stateful_pool`.
    """
    g.sess = tf.compat.v1.Session()
    g.sess.__enter__()


def worker_init_tf_vars(g):
    """Initialize the policy parameters on a worker.

    Parameters
    ----------
    g : :py:class:`garage.sampler.stateful_pool.SharedGlobal`
        SharedGlobal class from :py:mod:`garage.sampler.stateful_pool`.
    """
    g.sess.run(tf.compat.v1.global_variables_initializer())


class BatchSampler(BaseSampler):
    """Collects samples in parallel using a stateful pool of workers.

    Parameters
    ----------
    algo : :py:class:`garage.np.algos.base.RLAlgorithm`
        The algorithm.
    env : :py:class:`ast_toolbox.envs.ASTEnv`
        The environment.
    n_envs : int
        Number of parallel environments to run.
    open_loop : bool
        True if the simulation is open-loop, meaning that AST must generate all actions ahead of time, instead
        of being able to output an action in sync with the simulator, getting an observation back before
        the next action is generated. False to get interactive control, which requires that `blackbox_sim_state`
        is also False.
    batch_simulate : bool
        When in `obtain_samples` with `open_loop == True`, the sampler will call `self.sim.batch_simulate_paths` if
        `batch_simulate` is True, and `self.sim.simulate` if False.
    sim : :py:class:`ast_toolbox.simulators.ASTSimulator`
        The simulator wrapper, inheriting from `ast_toolbox.simulators.ASTSimulator`.
    reward_function : :py:class:`ast_toolbox.rewards.ASTReward`
        The reward function, inheriting from `ast_toolbox.rewards.ASTReward`.
    Args:
        algo (garage.np.algos.RLAlgorithm): The algorithm.
        env (gym.Env): The environment.

    """

    def __init__(self, algo, env, n_envs=1, open_loop=True, batch_simulate=False,
                 sim=ExampleAVSimulator(), reward_function=ExampleAVReward()):
        """


        """
        # pdb.set_trace()
        super(BatchSampler, self).__init__(algo, env)
        self.n_envs = n_envs
        self.open_loop = open_loop
        self.sim = sim
        self.reward_function = reward_function
        self.open_loop = open_loop
        self.batch_simulate = batch_simulate

    def start_worker(self):
        """Initialize the sampler."""
        assert singleton_pool.initialized, (
            'Use singleton_pool.initialize(n_parallel) to setup workers.')
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        parallel_sampler.populate_task(self.env, self.algo.policy)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf_vars)

    # def shutdown_worker(self):
    #     """Shutdown worker function."""
    #     parallel_sampler.terminate_task(scope=self.algo.scope)

    def shutdown_worker(self):
        """Terminate workers if necessary."""
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr, batch_size=None, whole_paths=True):
        """Collect samples for the given iteration number.

        Parameters
        ----------
        itr : int
            Iteration number.
        batch_size : int, optional
            How many simulation steps to run in each epoch.
        whole_paths : bool, optional
            Whether to return the full rollout paths data.
        """
        if not batch_size:
            batch_size = self.algo.max_path_length * self.n_envs

        # cur_params = self.algo.policy.get_param_values()
        cur_policy_params = self.algo.policy.get_param_values()
        cur_env_params = self.algo.env.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=cur_policy_params,
            max_samples=batch_size,
            max_path_length=self.algo.max_path_length,
            env_params=cur_env_params,
            scope=self.algo.scope,
        )
        # TODO: Doing the path correction here means the simulations will not be parallel.
        #  Need to make own parallel sampler and put it there to make that work
        if self.open_loop:
            if self.batch_simulate:
                # import pdb; pdb.set_trace()
                paths = self.sim.batch_simulate_paths(paths=paths, reward_function=self.reward_function)
            else:
                for path in paths:
                    s_0 = path["observations"][0]

                    # actions = path['env_infos']['info']['actions']
                    actions = path['actions']
                    # pdb.set_trace()
                    end_idx, info = self.sim.simulate(actions=actions, s_0=s_0)
                    # print('----- Back from simulate: ', end_idx)
                    if end_idx >= 0:
                        # pdb.set_trace()
                        self.slice_dict(path, end_idx)
                    rewards = self.reward_function.give_reward(
                        action=actions[end_idx],
                        info=self.sim.get_reward_info()
                    )
                    # print('----- Back from rewards: ', rewards)
                    # pdb.set_trace()
                    path["rewards"][end_idx] = rewards
                    # info[:, -1] = path["rewards"][:info.shape[0]]
                    # path['env_infos']['cache'] = info
                    path['env_infos']['cache'] = np.zeros_like(path["rewards"])
                    # import pdb; pdb.set_trace()

        # return paths if whole_paths else truncate_paths(paths, batch_size)
        if whole_paths:
            return paths
        else:
            paths_truncated = truncate_paths(paths, batch_size)
            return paths_truncated

    def slice_dict(self, in_dict, slice_idx):
        """Helper function to recursively parse through a dictionary of dictionaries and arrays to slice \
        the arrays at a certain index.

        Parameters
        ----------
        in_dict : dict
            Dictionary where the values are arrays or other dictionaries that follow this stipulation.
        slice_idx : int
            Index to slice each array at.

        Returns
        -------
        dict
            Dictionary where arrays at every level are sliced.

        """
        for key, value in in_dict.items():
            # pdb.set_trace()
            if isinstance(value, dict):
                in_dict[key] = self.slice_dict(value, slice_idx)
            else:
                in_dict[key][slice_idx + 1:, ...] = np.zeros_like(value[slice_idx + 1:, ...])

        return in_dict
