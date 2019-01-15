import time
from garage.algos.base import RLAlgorithm
import garage.misc.logger as logger
from garage.tf.policies.base import Policy
import tensorflow as tf
from garage.tf.samplers.batch_sampler import BatchSampler
# from garage.tf.samplers.vectorized_sampler import VectorizedSampler
from garage.sampler.utils import rollout
from garage.misc import ext
from garage.misc.overrides import overrides
import garage.misc.logger as logger
from garage.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from garage.tf.algos.batch_polopt import BatchPolopt
from garage.tf.misc import tensor_utils
import tensorflow as tf
import pdb
import numpy as np

from mylab.samplers.vectorized_is_sampler import VectorizedGASampler

class GAIS(BatchPolopt):
	"""
	Genetic Algorithm with Importance Sampling
	"""

	def __init__(
			self,
			top_paths = None,
			step_size=0.01, #serve as the std dev in mutation
			pop_size = 5,
			elites = 2,
			keep_best = 1,
			f_F = "max",
			**kwargs):

		self.top_paths = top_paths
		self.step_size = step_size
		self.pop_size = pop_size
		self.elites = elites
		self.f_F = f_F
		self.keep_best = keep_best
		self.sum_other_weights = np.zeros(pop_size)
		super(GAIS, self).__init__(**kwargs, sampler_cls=VectorizedISSampler)

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
			valid_var = None

		dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
		kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
		lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)

		input_list = [
						 obs_var,
						 action_var,
						 advantage_var,
					 ] + state_info_vars_list + old_dist_info_vars_list

		if is_recurrent:
			input_list.append(valid_var)

		self.f_lr = tensor_utils.compile_function(
				inputs=input_list,
				outputs=lr,
				log_name="f_lr",
			)
		return dict()

	@overrides
	def get_lr(self, samples_data):
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
		nenv, max_path_length, _ = all_input_values[0].shape 
		if not self.policy.recurrent:
			all_input_values_new = ()
			for (i,item) in enumerate(all_input_values):
				assert item.shape[0] == nenv
				assert item.shape[1] == max_path_length
				all_input_values_new += (np.reshape(item,(nenv*max_path_length,)+item.shape[2:]),)
			lr = self.f_lr(*all_input_values_new)
		else:
			lr = self.f_lr(*all_input_values)
		lr = np.reshape(lr,(nenv,max_path_length))
		return lr

	def train(self, sess=None, init_var=True):
		created_session = True if (sess is None) else False
		if sess is None:
			sess = tf.Session()
			sess.__enter__()
		if init_var:
			sess.run(tf.global_variables_initializer())
		self.start_worker()
		start_time = time.time()
		self.seeds = np.zeros([self.n_itr, self.pop_size])
		self.seeds[0,:] = np.random.randint(low= 0, high = int(2**16),
											size = (1, self.pop_size))
		for itr in range(self.n_itr):
			itr_start_time = time.time()
			with logger.prefix('itr #%d | ' % itr):
				all_paths = {}
				for p in range(self.pop_size):
					with logger.prefix('idv #%d | ' % p):
						logger.log("Updating Params")
						self.set_params(itr, p)

						logger.log("Obtaining samples...")
						paths = self.obtain_samples(itr)
						logger.log("Processing samples...")
						samples_data = self.process_samples(itr, paths)

						undiscounted_returns = [sum(path["rewards"]) for path in paths]

						if not (self.top_paths is None):
							action_seqs = [path["actions"] for path in paths]
							[self.top_paths.enqueue(action_seq,R,make_copy=True) for (action_seq,R) in zip(action_seqs,undiscounted_returns)]

						# all_paths[p]=paths
						all_paths[p]=samples_data

						logger.log("Logging diagnostics...")
						self.log_diagnostics(paths)
						logger.log("Saving snapshot...")
						snap = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
						if self.store_paths:
							snap["paths"] = samples_data["paths"]
						logger.save_itr_params(itr, snap)
						logger.log("Saved")
						logger.record_tabular('Itr',itr)
						logger.record_tabular('Ind',p)
						logger.record_tabular('StepNum',int(itr*self.batch_size*self.pop_size+self.batch_size*(p+1)))
						if self.top_paths is not None:
							for (topi, path) in enumerate(self.top_paths):
								logger.record_tabular('reward '+str(topi), path[0])
						logger.record_tabular('SumOtherWeights',self.sum_other_weights[p])

						logger.dump_tabular(with_prefix=False)

				logger.log("Optimizing Population...")
				self.optimize_policy(itr, all_paths)

		self.shutdown_worker()
		if created_session:
			sess.close()

	def set_params(self, itr, p):
		param_values = np.zeros_like(self.policy.get_param_values(trainable=True))
		for i in range(itr+1):
			# print("seed: ", self.seeds[i,p])
			if self.seeds[i,p] != 0:
				if i == 0:
					np.random.seed(int(self.seeds[i,p]))
					param_values = param_values + np.random.normal(size=param_values.shape)
				else:
					np.random.seed(int(self.seeds[i,p]))
					param_values = param_values + self.step_size*np.random.normal(size=param_values.shape)
		self.policy.set_param_values(param_values, trainable=True)

	def get_fitness(self, itr, all_paths):
		fitness = np.zeros(self.pop_size)
		for p in range(self.pop_size):
			self.set_params(itr,p)
			self.sum_other_weights[p] = 0.0
			for p_key in all_paths.keys():
				log_lrs = np.log(self.get_lr(all_paths[p_key]))
				valid_log_lrs = log_lrs*all_paths[p_key]["valids"]
				valid_log_lrs = log_lrs*all_paths[p_key]["valids"] #nan is from -inf*0.0
				valid_log_lrs[np.isnan(valid_log_lrs)] = 0.0 #set nan to 0.0 so won't influence sum
				path_lrs = np.exp(np.sum(valid_log_lrs,-1))
				if not p_key == p:
					self.sum_other_weights[p] += np.sum(path_lrs)

				rewards = all_paths[p_key]["rewards"]
				valid_rewards = rewards*all_paths[p_key]["valids"]
				path_rewards = np.sum(valid_rewards,-1)
				fitness[p] += np.sum(path_lrs*path_rewards)
		return fitness

	@overrides
	def optimize_policy(self, itr, all_paths):
		fitness = self.get_fitness(itr, all_paths)
		# print("fitness: ",fitness)
		sort_indx = np.flip(np.argsort(fitness),axis=0)

		new_seeds = np.zeros_like(self.seeds)
		for i in range(0, self.elites):
			new_seeds[:,i] = self.seeds[:,sort_indx[i]]
		for i in range(self.elites, self.pop_size):
			parent_idx = np.random.randint(low=0, high=self.elites)
			new_seeds[:,i] = new_seeds[:,parent_idx]
		if itr+1 < self.n_itr:
			new_seeds[itr+1, :] = np.random.randint(low= 0, high = int(2**16),
												size = (1, self.pop_size))
			for i in range(0,self.keep_best):
				new_seeds[itr+1,i] = 0

		self.seeds=new_seeds
		return dict()

	@overrides
	def get_itr_snapshot(self, itr, samples_data):
		# pdb.set_trace()
		return dict(
			itr=itr,
			policy=self.policy,
			seeds=self.seeds,
			env=self.env,
		)
