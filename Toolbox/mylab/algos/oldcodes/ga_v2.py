import time
from garage.algos.base import RLAlgorithm
import garage.misc.logger as logger
from garage.tf.policies.base import Policy
import tensorflow as tf
from garage.tf.samplers.batch_sampler import BatchSampler
from garage.tf.samplers.vectorized_sampler import VectorizedSampler
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

class GA(BatchPolopt):
	"""
	Genetic Algorithm 
	"""

	def __init__(
			self,
			policy = None,
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
		# self.init_param_values = policy.get_param_values(trainable=True)
		super(GA, self).__init__(**kwargs, policy=policy)

	@overrides
	def init_opt(self):
		return dict()

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
			fitness = np.zeros(self.pop_size)
			itr_start_time = time.time()
			with logger.prefix('itr #%d | ' % itr):
				for p in range(self.pop_size):
					with logger.prefix('idv #%d | ' % p):
						logger.log("Updating Params")

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
						# print("param values: ",self.policy.get_param_values(trainable=True))

						logger.log("Obtaining samples...")
						paths = self.obtain_samples(itr)
						logger.log("Processing samples...")
						samples_data = self.process_samples(itr, paths)
						undiscounted_returns = [sum(path["rewards"]) for path in paths]

						if not (self.top_paths is None):
							action_seqs = [path["actions"] for path in paths]
							[self.top_paths.enqueue(action_seq,R,make_copy=True) for (action_seq,R) in zip(action_seqs,undiscounted_returns)]

						if self.f_F == "max":
							fitness[p] = np.max(undiscounted_returns)
						else:
							fitness[p] = np.mean(undiscounted_returns)
						logger.log("Logging diagnostics...")
						self.log_diagnostics(paths)
						logger.log("Saving snapshot...")
						snap = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
						if self.store_paths:
							snap["paths"] = samples_data["paths"]
						logger.save_itr_params(itr, snap)
						logger.log("Saved")
						# logger.record_tabular('Time', time.time() - start_time)
						# logger.record_tabular('ItrTime', time.time() - itr_start_time)
						# logger.dump_tabular(with_prefix=False)
						logger.record_tabular('Itr',itr)
						logger.record_tabular('Ind',p)
						logger.record_tabular('StepNum',int(itr*self.batch_size*self.pop_size+self.batch_size*(p+1)))
						if self.top_paths is not None:
							for (topi, path) in enumerate(self.top_paths):
								logger.record_tabular('reward '+str(topi), path[0])

						logger.dump_tabular(with_prefix=False)

				# logger.log("Logging diagnostics...")
				# self.log_diagnostics(paths)
				logger.log("Optimizing Population...")
				self.optimize_policy(itr, fitness)
				# logger.log("Saving snapshot...")
				# params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
				# if self.store_paths:
				#     params["paths"] = samples_data["paths"]
				# logger.save_itr_params(itr, params)
				# logger.log("Saved")

				# if self.plot:
				#     rollout(self.env, self.policy, animated=True, max_path_length=self.max_path_length)
				#     if self.pause_for_plot:
				#         input("Plotting evaluation run: Press Enter to "
				#               "continue...")
		self.shutdown_worker()
		if created_session:
			sess.close()

	@overrides
	def optimize_policy(self, itr, fitness):
		# pdb.set_trace()
		sort_indx = np.flip(np.argsort(fitness),axis=0)

		# print("old_seeds: ",self.seeds[:itr+1,:])
		# print("fitness: ",fitness)
		# print("sort_index: ",sort_indx)

		new_seeds = np.zeros_like(self.seeds)
		for i in range(0, self.elites):
			new_seeds[:,i] = self.seeds[:,sort_indx[i]]
			# print("assign elites ",i)
			# print("value extracted: ",self.seeds[:,sort_indx[i]])
			# print("new_seeds: ",new_seeds[:itr+1,:])
		for i in range(self.elites, self.pop_size):
			parent_idx = np.random.randint(low=0, high=self.elites)
			new_seeds[:,i] = new_seeds[:,parent_idx]
			# print("assign others ",i)
			# print("parent_idx: ",parent_idx)
			# print("value extracted: ",new_seeds[:,parent_idx])
			# print("new_seeds: ",new_seeds[:itr+1,:])
		if itr+1 < self.n_itr:
			new_seeds[itr+1, :] = np.random.randint(low= 0, high = int(2**16),
												size = (1, self.pop_size))
			for i in range(0,self.keep_best):
				new_seeds[itr+1,i] = 0

		self.seeds=new_seeds

		# print("new_seeds: ",new_seeds[:itr+2,:])
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
