# from garage.tf.policies.base2 import StochasticPolicy2
import numpy as np
import tensorflow as tf
from garage.tf.distributions.diagonal_gaussian import DiagonalGaussian
from garage.tf.policies.base import StochasticPolicy


class GoExplorePolicy(StochasticPolicy):
    def __init__(self, env_spec, name='GoExplorePolicy'):
        self.dist = DiagonalGaussian(dim=env_spec.action_space.flat_dim)
        self.log_std = np.zeros(env_spec.action_space.flat_dim)
        # self.cell_num = 0
        # self.stateful_num = -2
        # self.cell_pool = None

        super(GoExplorePolicy, self).__init__(env_spec=env_spec, name=name)
        self._initialize()

    def _initialize(self):
        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs

    # Should be implemented by all policies
    def get_action(self, observation):
        # print("From get_action: ", self, ": ", self.cell_num, ", ", self.action_iter)
        # self.action_iter += 1
        # if self.action_iter <= self.cell.trajectory_length:
        #     return self.cell.trajectory[self.action_iter]
        return self.action_space.sample(), dict(mean=self.log_std, log_std=self.log_std)

    def get_actions(self, observations):
        # import pdb; pdb.set_trace()
        # obs = [path["observations"] for path in paths]
        means = [self.log_std for observation in observations]
        log_stds = [self.log_std for observation in observations]
        return self.action_space.sample_n(len(observations)), dict(mean=means, log_std=log_stds)

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        # print("From reset: ", self, ": ", self.cell_num, ", ", self.stateful_num, ": ", self.cell_pool)
        # import pdb; pdb.set_trace()
        # self.cell_num = np.random.randint(0,self.cell_pool.length)
        # self.action_iter =-1
        # self.cell = self.cell_pool.get_cell(self.cell_num)
        # print("cell pool length from ", self, ": ", len(self.cell_pool.pool))
        # print("reset policy")
        pass

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """

    @property
    def vectorized(self):
        """
        Indicates whether the policy is vectorized. If True, it should
        implement get_actions(), and support resetting
        with multiple simultaneous states.
        """
        return False

    def terminate(self):
        """
        Clean up operation
        """

    @property
    def distribution(self):
        return self.dist

    def dist_info(self, obs, state_infos):
        """
        Distribution info.

        Return the distribution information about the actions.
        :param obs_var: observation values
        :param state_info_vars: a dictionary whose values should contain
         information about the state of the policy at the time it received the
         observation
        :return:
        """
        return dict(mean=None, log_std=self.log_std)

    def dist_info_sym(self, obs_var, state_info_vars, name='dist_info_sym'):
        raise NotImplementedError

    # def get_param_values(self):
    #     print("params ", self.cell_num, ", ", self.stateful_num, ", ", self.cell_pool, " retrieved from ", self)
    #     return {"cell_num": self.cell_num,
    #             "stateful_num": self.stateful_num,
    #             "cell_pool": self.cell_pool}
    #
    # def set_param_values(self, params):
    #     self.cell_num = params["cell_num"]
    #     self.stateful_num = params["stateful_num"]
    #     self.cell_pool = params["cell_pool"]
    #     print(self, " had params set to ", self.cell_num, ", ", self.stateful_num)
    #
    # def set_cell_pool(self, cell_pool):
    #     self.cell_pool = cell_pool
    #     print(self, "had cell pool set to: ", self.cell_pool)
