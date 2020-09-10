import numpy as np
import tensorflow as tf
from garage.tf.distributions.diagonal_gaussian import DiagonalGaussian
from garage.tf.policies.base import StochasticPolicy


class GoExplorePolicy(StochasticPolicy):
    """A stochastic policy for Go-Explore that takes actions uniformally at random.

    Parameters
    ----------
    env_spec : :py:class:`garage.envs.EnvSpec`
        Environment specification.
    name : str
        Name for the tensors.
    """

    def __init__(self, env_spec, name='GoExplorePolicy'):

        self.dist = DiagonalGaussian(dim=env_spec.action_space.flat_dim)
        self.log_std = np.zeros(env_spec.action_space.flat_dim)

        super(GoExplorePolicy, self).__init__(env_spec=env_spec, name=name)
        self._initialize()

    def _initialize(self):
        """Initialize the tensor variable scope.

        """
        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs

    # Should be implemented by all policies
    def get_action(self, observation):
        """Get action sampled from the policy.

        Parameters
        ----------
        observation : array_like
            Observation from the environment.

        Returns
        -------
        array_like
            Action sampled from the policy.

        """
        return self.action_space.sample(), dict(mean=self.log_std, log_std=self.log_std)

    def get_actions(self, observations):
        """Get actions sampled from the policy.

        Parameters
        ----------
        observations : list[array_like]
            Observations from the environment.

        Returns
        -------
        array_like
            Actions sampled from the policy.

        """
        means = [self.log_std for observation in observations]
        log_stds = [self.log_std for observation in observations]
        return self.action_space.sample_n(len(observations)), dict(mean=means, log_std=log_stds)

    # def get_params_internal(self, **tags):
    #     """
    #
    #     Parameters
    #     ----------
    #     tags :
    #
    #     Returns
    #     -------
    #
    #     """
    #     return []

    def reset(self, dones=None):
        """Reset the policy.

        If dones is None, it will be by default np.array([True]) which implies
        the policy will not be "vectorized", i.e. number of parallel
        environments for training data sampling = 1.

        Parameters
        ----------
        dones : array_like
            Bool that indicates terminal state(s).
        """

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths.
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
        Clean up operation.
        """

    @property
    def distribution(self):
        """Distribution.

        Returns
        -------

            Distribution.
        """
        return self.dist

    def dist_info(self, obs, state_infos):
        """
        Distribution info.

        Return the distribution information about the actions.

        Parameters
        ----------
        obs : array_like
             Observation values.
        state_infos : dict
            A dictionary whose values should contain
            information about the state of the policy at the time it received the
            observation.
        """
        return dict(mean=None, log_std=self.log_std)

    def dist_info_sym(self, obs_var, state_info_vars, name='dist_info_sym'):
        """Symbolic graph of the distribution.

        Return the symbolic distribution information about the actions.

        Parameters
        ----------
        obs_var : `tf.Tensor <https://www.tensorflow.org/api_docs/python/tf/Tensor>`_
            Symbolic variable for observations.
        state_infos : dict
            A dictionary whose values should contain
            information about the state of the policy at the time it received the
            observation.
        name : str
            Name of the symbolic graph.
        """
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
