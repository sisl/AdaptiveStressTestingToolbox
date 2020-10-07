import numpy as np
import tensorflow as tf
from dowel import tabular
from garage.tf.misc import tensor_utils

from ast_toolbox.algos import GA
from ast_toolbox.optimizers import DirectionConstraintOptimizer


class GASM(GA):
    """Deep Genetic Algorithm [1]_ with Safe Mutation [2]_.

    Parameters
    ----------
    step_size : float, optional
        The constraint on the KL divergence of each mutation.
    kwargs :
        Keyword arguments passed to `ast_toolbox.algos.ga.GA`.

    References
    ----------
    .. [1] Such, Felipe Petroski, et al. "Deep neuroevolution: Genetic algorithms are a competitive alternative for
    training deep neural networks for reinforcement learning."
     arXiv preprint arXiv:1712.06567 (2017).
    .. [2] Lehman, Joel, et al. "Safe mutations for deep and recurrent neural networks through output gradients."
     Proceedings of the Genetic and Evolutionary Computation Conference. 2018.
    """

    def __init__(
            self,
            step_size=0.01,
            **kwargs):

        self.divergences = np.zeros(kwargs['pop_size'])
        self.optimizer = DirectionConstraintOptimizer()
        super(GASM, self).__init__(**kwargs, step_size=step_size)

    def init_opt(self):
        """Initiate trainer internal tensorflow operations.
        """
        is_recurrent = int(self.policy.recurrent)
        # obs_var = self.env_spec.observation_space.new_tensor_variable(
        #     'obs',
        #     extra_dims=1 + is_recurrent,
        # )
        # action_var = self.env_spec.action_space.new_tensor_variable(
        #     'action',
        #     extra_dims=1 + is_recurrent,
        # )
        if is_recurrent:
            obs_var = tf.compat.v1.placeholder(
                tf.float32,
                shape=[None, None, self.env_spec.observation_space.flat_dim],
                name='obs')
            action_var = tf.compat.v1.placeholder(
                tf.float32,
                shape=[None, None, self.env_spec.action_space.flat_dim],
                name='obs')
        else:
            obs_var = tf.compat.v1.placeholder(
                tf.float32,
                shape=[None, self.env_spec.observation_space.flat_dim],
                name='obs')
            action_var = tf.compat.v1.placeholder(
                tf.float32,
                shape=[None, self.env_spec.action_space.flat_dim],
                name='obs')

        # advantage_var = tensor_utils.new_tensor(
        #     'advantage',
        #     ndim=1 + is_recurrent,
        #     dtype=tf.float32,
        # )

        state_info_vars = {
            k: tf.compat.v1.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.compat.v1.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = tf.compat.v1.placeholder(tf.float32, shape=[None], name="valid")

        # npath_var = tf.compat.v1.placeholder(tf.int32, shape=(), name="npath")
        npath_var = tf.compat.v1.placeholder(tf.int32, shape=[None], name="npath")  # in order to work with sliced_fn

        actions = self.policy.get_action_sym(obs_var, name='policy_action')
        divergence = tf.reduce_sum(
            tf.reduce_sum(tf.square(actions - action_var), -1) * valid_var) / tf.reduce_sum(valid_var)

        input_list = [
            obs_var,
            action_var,
            # advantage_var,
        ] + state_info_vars_list

        input_list.append(valid_var)
        input_list.append(npath_var)

        self.f_divergence = tensor_utils.compile_function(
            inputs=input_list,
            outputs=divergence,
            log_name="f_divergence",
        )

        self.optimizer.update_opt(
            target=self.policy,
            # leq_constraint=(mean_kl, self.step_size),
            leq_constraint=divergence,  # input max constraint at run time with annealing
            inputs=input_list,
            constraint_name="divergence"
        )
        return dict()

    def extra_recording(self, itr):
        """Record extra training statistics per-iteration.

        Parameters
        ----------
        itr : int
            The iteration number.
        """
        tabular.record('Max Divergence', np.max(self.divergences))
        tabular.record('Min Divergence', np.min(self.divergences))
        tabular.record('Mean Divergence', np.mean(self.divergences))
        return None

    def data2inputs(self, samples_data):
        """Transfer the processed data samples to training inputs

        Parameters
        ----------
        samples_data : dict
            The processed data samples

        Returns
        -------
        all_input_values : tuple
            The input used in training
        """

        all_input_values = (samples_data["observations"], samples_data["actions"])  # ,samples_data["advantages"])
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        all_input_values += tuple(state_info_list)
        # if self.policy.recurrent:
        all_input_values += (samples_data["valids"],)
        npath, max_path_length, _ = all_input_values[0].shape
        if not self.policy.recurrent:
            all_input_values_new = ()
            for (i, item) in enumerate(all_input_values):
                assert item.shape[0] == npath
                assert item.shape[1] == max_path_length
                all_input_values_new += (np.reshape(item, (npath * max_path_length,) + item.shape[2:]),)
            all_input_values_new += (np.ones(npath * max_path_length,) * npath,)
            return all_input_values_new
        else:
            all_input_values += (np.ones(npath) * npath,)
        return all_input_values

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
        self.seeds = np.copy(new_seeds)
        self.magnitudes = np.copy(new_magnitudes)
        new_seeds[itr + 1, :] = np.random.randint(low=0, high=int(2**16),
                                                  size=(1, self.pop_size))
        for i in range(0, self.keep_best):
            new_seeds[itr + 1, i] = 0
        for p in range(self.pop_size):
            self.set_params(itr, p)
            param_values = self.policy.get_param_values(trainable=True)

            # np.random.seed(int(new_seeds[itr+1,p]))
            # direction = np.random.normal(size=param_values.shape)
            self.np_random.seed(int(new_seeds[itr + 1, p]))
            direction = self.np_random.normal(size=param_values.shape)

            samples_data = all_paths[self.parents[p]]
            all_input_values = self.data2inputs(samples_data)

            new_magnitudes[itr + 1, p], constraint_val = \
                self.optimizer.get_magnitude(
                    direction=direction, inputs=all_input_values, max_constraint_val=self.step_size)
            self.divergences[p] = constraint_val
        return new_seeds, new_magnitudes

    def __getstate__(self):
        """
        Get the internal state.

        Returns
        -------
        data : dict
            The intertal state dict.
        """
        data = self.__dict__.copy()
        del data['f_divergence']
        return data

    def __setstate__(self, state):
        """Set the internal state."""
        self.__dict__ = state
        self._name_scope = tf.name_scope(self.name)
        self.init_opt()

    # for debug
    # def optimize_policy(self, itr, all_paths):
    # 	fitness = self.get_fitness(itr, all_paths)
    # 	self.select_parents(fitness)
    # 	new_seeds = np.zeros_like(self.seeds)
    # 	new_seeds[:,:] = self.seeds[:,self.parents]
    # 	new_magnitudes = np.zeros_like(self.magnitudes)
    # 	new_magnitudes[:,:] = self.magnitudes[:,self.parents]
    # 	if itr+1 < self.n_itr:
    # 		new_seeds, new_magnitudes = self.mutation(itr, new_seeds, new_magnitudes, all_paths)
    # 	self.seeds=new_seeds
    # 	self.magnitudes=new_magnitudes
    # 	# print(self.seeds)
    # 	# print(self.magnitudes)
    # 	for p in range(self.pop_size):
    # 		self.set_params(itr+1,p)
    # 		p_key = self.parents[p]
    # 		all_input_values = self.data2inputs(all_paths[p_key])
    # 		divergence = self.f_divergence(*all_input_values)
    # 		print(divergence)
    # 		self.divergences[p] = divergence
    # 	return dict()
