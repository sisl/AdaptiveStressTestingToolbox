import time
# from garage.tf.samplers.vectorized_sampler import VectorizedSampler
import garage.misc.logger as logger
from garage.tf.algos.batch_polopt import BatchPolopt
import tensorflow as tf
import numpy as np

from ast_toolbox.samplers.vectorized_ga_sampler import VectorizedGASampler
from ast_toolbox.utils import seeding


class GA(BatchPolopt):
	"""
	Genetic Algorithm
	"""

	def __init__(
			self,
			top_paths = None, 
			step_size = 0.01,
			step_size_anneal = 1.0,
			pop_size = 5,
			truncation_size = 2,
			keep_best = 1,
			f_F = "mean",
			log_interval = 4000,
			init_step = 1.0,
			**kwargs):
		"""
		:param top_paths: a bounded priority queue to store top-rewarded trajectories
		:param step_size: standard deviation for each mutation
		:param step_size_anneal: the linear annealing rate of step_size after each iteration
		:param pop_size: the population size
		:param truncation_size: the number of top-performed individuals that are chosen as parents
		:param keep_best: the number of top-performed individuals that remain unchanged for next generation
		:param f_F: the function used to calculate fitness: 'mean' for the average return, 'max' for the max return
		:param log_interval: the log interval in terms of environment calls
		:return: No return value.
		"""

		self.top_paths = top_paths
		self.best_mean = -np.inf
		self.best_var = 0.0
		self.step_size = step_size
		self.step_size_anneal = step_size_anneal
		self.pop_size = pop_size
		self.truncation_size = truncation_size
		self.keep_best = keep_best
		self.f_F = f_F
		self.log_interval = log_interval
		self.init_step = init_step

		self.seeds = np.zeros([kwargs['n_itr'], pop_size],dtype=int)
		self.magnitudes = np.zeros([kwargs['n_itr'], pop_size])
		self.parents = np.zeros(pop_size,dtype=int)
		self.np_random, seed = seeding.np_random() #used in set_params
		super(GA, self).__init__(**kwargs, sampler_cls=VectorizedGASampler)
		
	def initial(self):
		self.seeds[0,:] = np.random.randint(low= 0, high = int(2**16),
											size = (1, self.pop_size))
		self.magnitudes[0,:] = self.init_step*np.ones(self.pop_size)
		self.policy.set_param_values(self.policy.get_param_values())
		self.stepNum = 0

	def init_opt(self):
		return dict()

	def train(self, sess=None, init_var=False):
		created_session = True if (sess is None) else False
		if sess is None:
			sess = tf.Session()
			sess.__enter__()
		if init_var:
			sess.run(tf.global_variables_initializer())
		self.start_worker(sess)
		start_time = time.time()
		self.initial()

		for itr in range(self.n_itr):
			itr_start_time = time.time()
			with logger.prefix('itr #%d | ' % itr):
				all_paths = {}
				for p in range(self.pop_size):
					with logger.prefix('idv #%d | ' % p):
						logger.log("Updating Params")
						self.set_params(itr, p)
						# print(self.policy.get_param_values(trainable=True))
						logger.log("Obtaining samples...")
						paths = self.obtain_samples(itr)
						logger.log("Processing samples...")
						samples_data = self.process_samples(itr, paths)
						# print([np.mean(path["actions"],-1) for path in paths])

						# all_paths[p]=paths
						all_paths[p]=samples_data

						logger.log("Logging diagnostics...")
						self.log_diagnostics(paths)
						# logger.log("Saving snapshot...")
						# snap = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
						# if self.store_paths:
						# 	snap["paths"] = samples_data["paths"]
						# logger.save_itr_params(itr, snap)
						# logger.log("Saved")

						self.record_tabular(itr,p)

				logger.log("Optimizing Population...")
				self.optimize_policy(itr, all_paths)
				self.step_size = self.step_size*self.step_size_anneal

		self.shutdown_worker()
		if created_session:
			sess.close()

	def record_tabular(self, itr, p):
		if self.stepNum%self.log_interval == 0:
			logger.record_tabular('Itr',itr)
			logger.record_tabular('Ind',p)
			logger.record_tabular('StepNum',self.stepNum)
			if self.top_paths is not None:
				for (topi, path) in enumerate(self.top_paths):
					logger.record_tabular('reward '+str(topi), path[0])
			logger.record_tabular('BestMean', self.best_mean)
			logger.record_tabular('BestVar', self.best_var)
			logger.record_tabular('parent',self.parents[p])
			logger.record_tabular('StepSize',self.step_size)
			logger.record_tabular('Magnitude',self.magnitudes[itr,p])
			self.extra_recording(itr, p)
			logger.dump_tabular(with_prefix=False)

	def extra_recording(self, itr, p):
		return None

	def set_params(self, itr, p):
		for i in range(itr+1):
			# print("seed: ", self.seeds[i,p])
			self.np_random.seed(int(self.seeds[i,p]))
			if i == 0: #first generation
				param_values = self.policy.get_param_values(trainable=True)
				param_values = self.magnitudes[i,p]*self.np_random.normal(size=param_values.shape)

				# param_values = init_policy_np(self.policy, self.np_random)

				# params = self.policy.get_params()
				# sess = tf.get_default_session()
				# sess.run(tf.variables_initializer(params))
				# param_values = self.policy.get_param_values()
			elif self.seeds[i,p] != 0:
				param_values = param_values + self.magnitudes[i,p]*self.np_random.normal(size=param_values.shape)
		self.policy.set_param_values(param_values, trainable=True)

	def get_fitness(self, itr, all_paths):
		fitness = np.zeros(self.pop_size)
		for p in range(self.pop_size):
			rewards = all_paths[p]["rewards"]
			valid_rewards = rewards*all_paths[p]["valids"]
			path_rewards = np.sum(valid_rewards,-1)
			if self.f_F == "max":
				fitness[p] = np.max(path_rewards)
			else:
				fitness[p] = np.mean(path_rewards)
		return fitness

	def select_parents(self, fitness):
		sort_indx = np.flip(np.argsort(fitness),axis=0)
		self.parents[0:self.truncation_size] = sort_indx[0:self.truncation_size]
		self.parents[self.truncation_size:self.pop_size] = \
				sort_indx[np.random.randint(low=0,high=self.truncation_size,size=self.pop_size-self.truncation_size)]

	def mutation(self, itr, new_seeds, new_magnitudes, all_paths):
		if itr+1 < self.n_itr:
			new_seeds[itr+1, :] = np.random.randint(low= 0, high = int(2**32),
												size = (1, self.pop_size))
			new_magnitudes[itr+1,: ] = self.step_size
			for i in range(0,self.keep_best):
				new_seeds[itr+1,i] = 0
		return new_seeds, new_magnitudes

	def optimize_policy(self, itr, all_paths):
		fitness = self.get_fitness(itr, all_paths)
		self.select_parents(fitness)
		new_seeds = np.zeros_like(self.seeds)
		new_seeds[:,:] = self.seeds[:,self.parents]
		new_magnitudes = np.zeros_like(self.magnitudes)
		new_magnitudes[:,:] = self.magnitudes[:,self.parents]
		if itr+1 < self.n_itr:
			new_seeds, new_magnitudes = self.mutation(itr, new_seeds, new_magnitudes, all_paths)
		self.seeds=new_seeds
		self.magnitudes=new_magnitudes
		return dict()

	def obtain_samples(self, itr):
		self.stepNum += self.batch_size
		paths = self.sampler.obtain_samples(itr)
		undiscounted_returns = [sum(path["rewards"]) for path in paths]
		if np.mean(undiscounted_returns) > self.best_mean:
			self.best_mean = np.mean(undiscounted_returns)
			self.best_var = np.var(undiscounted_returns)
		if not (self.top_paths is None):
			action_seqs = [path["actions"] for path in paths]
			[self.top_paths.enqueue(action_seq,R,make_copy=True) for (action_seq,R) in zip(action_seqs,undiscounted_returns)]
		return paths

	def get_itr_snapshot(self, itr, samples_data):
		# pdb.set_trace()
		return dict(
			itr=itr,
			policy=self.policy,
			seeds=self.seeds,
			env=self.env,
		)
