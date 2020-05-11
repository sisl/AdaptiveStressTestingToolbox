# import tensorflow as tf
# from ast_toolbox.samplers import parallel_sampler
# from garage.sampler import singleton_pool
# from garage.sampler.base import BaseSampler
# from garage.sampler.utils import truncate_paths
#
#
# def worker_init_tf(g):
#     g.sess = tf.compat.v1.Session()
#     g.sess.__enter__()
#
#
# def worker_init_tf_vars(g):
#     g.sess.run(tf.compat.v1.global_variables_initializer())
#
#
# class BatchSampler(BaseSampler):
#     def __init__(self, algo, env, n_envs):
#         super(BatchSampler, self).__init__(algo, env)
#         self.n_envs = n_envs
#
#     def start_worker(self):
#         assert singleton_pool.initialized, (
#             'Use singleton_pool.initialize(n_parallel) to setup workers.')
#         if singleton_pool.n_parallel > 1:
#             singleton_pool.run_each(worker_init_tf)
#         parallel_sampler.populate_task(self.env, self.algo.policy)
#         if singleton_pool.n_parallel > 1:
#             singleton_pool.run_each(worker_init_tf_vars)
#
#     def shutdown_worker(self):
#         parallel_sampler.terminate_task(scope=self.algo.scope)
#
#     def obtain_samples(self, itr, batch_size=None, whole_paths=True):
#         if not batch_size:
#             batch_size = self.algo.max_path_length * self.n_envs
#
#         cur_policy_params = self.algo.policy.get_param_values()
#         cur_env_params = self.algo.env.get_param_values()
#         paths = parallel_sampler.sample_paths(
#             policy_params=cur_policy_params,
#             max_samples=batch_size,
#             max_path_length=self.algo.max_path_length,
#
#             scope=self.algo.scope,
#         )
#         if whole_paths:
#             return paths
#         else:
#             paths_truncated = truncate_paths(paths, batch_size)
#             return paths_truncated
from ast_toolbox.samplers import parallel_sampler
from garage.sampler.base import BaseSampler
from garage.sampler.utils import truncate_paths


class BatchSampler(BaseSampler):
    """Class with batch-based sampling.

    Args:
        algo (garage.np.algos.RLAlgorithm): The algorithm.
        env (gym.Env): The environment.

    """

    def __init__(self, algo, env):
        super(BatchSampler, self).__init__(algo, env)

    def start_worker(self):
        """Start worker function."""
        parallel_sampler.populate_task(
            self.env, self.algo.policy, scope=self.algo.scope)

    def shutdown_worker(self):
        """Shutdown worker function."""
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr, batch_size=None, whole_paths=True):
        """Obtain samples function."""
        if not batch_size:
            batch_size = self.algo.max_path_length

        cur_params = self.algo.policy.get_param_values()
        cur_env_params = self.algo.env.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=batch_size,
            max_path_length=self.algo.max_path_length,
            env_params=cur_env_params,
            scope=self.algo.scope,
        )

        return paths if whole_paths else truncate_paths(paths, batch_size)
