import garage.misc.logger as logger
import tensorflow as tf
from garage.misc import ext
import garage.misc.logger as logger
from garage.tf.misc import tensor_utils
import tensorflow as tf
import numpy as np

from ast_toolbox.algos.ga import GA
from ast_toolbox.optimizers.direction_constraint_optimizer import DirectionConstraintOptimizer

class GASM(GA):
	"""
	Genetic Algorithm with Safe Mutation
	"""

	def __init__(
			self,
			optimizer=None,
			**kwargs):
		"""
		:param step_size: the constraint on the KL divergence of each mutation
		"""
		self.divergences = np.zeros(kwargs['pop_size'])
		if optimizer == None:
			self.optimizer = DirectionConstraintOptimizer()
		else:
			self.optimizer = optimizer
		super(GASM, self).__init__(**kwargs)

	def init_opt(self):
		is_recurrent = int(self.policy.recurrent)
		obs_var = self.env.observation_space.new_tensor_variable(
			'obs',
			extra_dims=1 + is_recurrent,
		)
		action_var = self.env.action_space.new_tensor_variable(
			'action',
			extra_dims=1 + is_recurrent,
		)
		advantage_var = tensor_utils.new_tensor(
			'advantage',
			ndim=1 + is_recurrent,
			dtype=tf.float32,
		)

		state_info_vars = {
			k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
			for k, shape in self.policy.state_info_specs
			}
		state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

		if is_recurrent:
			valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
		else:
			valid_var = tf.placeholder(tf.float32, shape=[None], name="valid")

		# npath_var = tf.placeholder(tf.int32, shape=(), name="npath") 
		npath_var = tf.placeholder(tf.int32, shape=[None], name="npath") #in order to work with sliced_fn

		actions = self.policy.get_action_sym(obs_var)
		divergence = tf.reduce_sum(tf.reduce_sum(tf.square(actions -  action_var),-1)*valid_var)/tf.reduce_sum(valid_var)

		input_list = [
						 obs_var,
						 action_var,
						 advantage_var,
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
			leq_constraint = divergence, #input max contraint at run time with annealing
			inputs=input_list,
			constraint_name="divergence"
		)
		return dict()

	def extra_recording(self, itr, p):
		logger.record_tabular('Divergence',self.divergences[p])
		return None

	def data2inputs(self, samples_data):
		all_input_values = tuple(ext.extract(
			samples_data,
			"observations", "actions", "advantages"
		))
		agent_infos = samples_data["agent_infos"]
		state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
		all_input_values += tuple(state_info_list)
		# if self.policy.recurrent:
		all_input_values += (samples_data["valids"],)
		npath, max_path_length, _ = all_input_values[0].shape 
		if not self.policy.recurrent:
			all_input_values_new = ()
			for (i,item) in enumerate(all_input_values):
				assert item.shape[0] == npath
				assert item.shape[1] == max_path_length
				all_input_values_new += (np.reshape(item,(npath*max_path_length,)+item.shape[2:]),)
			all_input_values_new += (np.ones(npath*max_path_length,)*npath,)
			return all_input_values_new
		else:
			all_input_values += (np.ones(npath)*npath,)
		return all_input_values

	def mutation(self, itr, new_seeds, new_magnitudes, all_paths):
		self.seeds=np.copy(new_seeds)
		self.magnitudes=np.copy(new_magnitudes)
		new_seeds[itr+1, :] = np.random.randint(low= 0, high = int(2**16),
											size = (1, self.pop_size))
		for i in range(0,self.keep_best):
			new_seeds[itr+1,i] = 0
		for p in range(self.pop_size):
			self.set_params(itr,p)
			param_values = self.policy.get_param_values(trainable=True)

			# np.random.seed(int(new_seeds[itr+1,p]))
			# direction = np.random.normal(size=param_values.shape)
			self.np_random.seed(int(new_seeds[itr+1,p]))
			direction = self.np_random.normal(size=param_values.shape)

			samples_data = all_paths[self.parents[p]]
			all_input_values = self.data2inputs(samples_data)

			new_magnitudes[itr+1,p], constraint_val = \
				self.optimizer.get_magnitude(direction=direction,inputs=all_input_values,max_constraint_val=self.step_size)
			self.divergences[p] = constraint_val
		return new_seeds, new_magnitudes

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

