import garage.misc.logger as logger
import tensorflow as tf
from garage.misc import ext
from garage.misc.overrides import overrides
import garage.misc.logger as logger
from garage.tf.misc import tensor_utils
import tensorflow as tf
import numpy as np

from mylab.algos.ga import GA
from mylab.optimizers.direction_constraint_optimizer import DirectionConstraintOptimizer

class GATR(GA):
	"""
	Genetic Algorithm with Trust Region Mutation
	"""

	def __init__(
			self,
			optimizer=None,
			**kwargs):

		self.kls = np.zeros(kwargs['pop_size'])
		if optimizer == None:
			self.optimizer = DirectionConstraintOptimizer()
		else:
			self.optimizer = optimizer
		super(GATR, self).__init__(**kwargs)

	@overrides
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
		dist = self.policy.distribution

		old_dist_info_vars = {
			k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
			for k, shape in dist.dist_info_specs
			}
		old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

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

		dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
		kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
		lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)

		mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
		log_lrs = tf.log(lr)
		valid_log_lrs = log_lrs*valid_var #nan is from -inf*0.0
		valid_log_lrs = tf.where(tf.is_nan(valid_log_lrs),tf.zeros_like(valid_log_lrs),valid_log_lrs) #set nan to 0.0 so won't influence sum
		valid_log_lrs = tf.reshape(valid_log_lrs,tf.stack([npath_var[0],-1]))
		path_lrs = tf.exp(tf.reduce_sum(valid_log_lrs,-1))

		input_list = [
						 obs_var,
						 action_var,
						 advantage_var,
					 ] + state_info_vars_list + old_dist_info_vars_list

		input_list.append(valid_var)
		input_list.append(npath_var)

		self.f_path_lrs = tensor_utils.compile_function(
				inputs=input_list,
				outputs=path_lrs,
				log_name="f_path_lrs",
			)

		self.f_mean_kl = tensor_utils.compile_function(
				inputs=input_list,
				outputs=mean_kl,
				log_name="f_mean_kl",
			)

		self.optimizer.update_opt(
			target=self.policy,
			# leq_constraint=(mean_kl, self.step_size), 
			leq_constraint = mean_kl, #input max contraint at run time with annealing
			inputs=input_list,
			constraint_name="mean_kl"
		)
		return dict()

	@overrides
	def extra_recording(self, itr):
		logger.record_tabular('Mean KL',np.mean(self.kls))
		logger.record_tabular('Std KL',np.std(self.kls))
		logger.record_tabular('Max KL',np.max(self.kls))
		logger.record_tabular('Min KL',np.min(self.kls))
		return None

	def data2inputs(self, samples_data):
		all_input_values = tuple(ext.extract(
			samples_data,
			"observations", "actions", "advantages"
		))
		agent_infos = samples_data["agent_infos"]
		state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
		dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
		all_input_values += tuple(state_info_list) + tuple(dist_info_list)
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

	@overrides
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
			self.np_randm.seed(int(new_seeds[itr+1,p]))
			direction = self.np_random.normal(size=param_values.shape)

			samples_data = all_paths[self.parents[p]]
			all_input_values = self.data2inputs(samples_data)

			new_magnitudes[itr+1,p], constraint_val = \
				self.optimizer.get_magnitude(direction=direction,inputs=all_input_values,max_constraint_val=self.step_size)
			self.kls[p] = constraint_val
		return new_seeds, new_magnitudes

	# @overrides
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
	# 	print(self.seeds)
	# 	print(self.magnitudes)
	# 	for p in range(self.pop_size):
	# 		self.set_params(itr+1,p)
	# 		p_key = self.parents[p]
	# 		all_input_values = self.data2inputs(all_paths[p_key])
	# 		mean_kl = self.f_mean_kl(*all_input_values)
	# 		print(mean_kl)
	# 		self.kls[p] = mean_kl
	# 	return dict()

