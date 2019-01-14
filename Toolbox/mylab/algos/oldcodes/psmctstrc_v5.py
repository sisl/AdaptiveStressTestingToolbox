from mylab.algos.psmctstr import PSMCTSTR
from mylab.optimizers.direction_constraint_optimizer import DirectionConstraintOptimizer
from mylab.utils.mcts_utils import *
from garage.misc.overrides import overrides
from garage.misc import ext
from garage.tf.misc import tensor_utils
import tensorflow as tf
import numpy as np

class PSMCTSTRC(PSMCTSTR):
	"""
	Policy Space MCTS with Trust Region Mutation and 
	candidate new actions: when adding a new mutation action, instead of just adding one, also adding multiple
			candidate actions, this is more data efficient since they share the parent trajectory to calculate 
			divergence.
	"""
	def __init__(
			self,
			n_ca=4,
			**kwargs):
		self.n_ca = n_ca
		super(PSMCTSTRC, self).__init__(**kwargs)

	@overrides
	def getNextAction(self,s):
		if len(self.s[s].ca) == 0:
			self.s[s].ca = self.getCandidateActions(s)
		a = self.s[s].ca.pop()
		return a

	def getCandidateActions(self,s,samples_data=None):
		actions = []
		if samples_data is None:
			self.set_params(s)
			paths = self.obtain_samples(0)
			samples_data = self.process_samples(0, paths)
		all_input_values = self.data2inputs(samples_data)
		# seeds = np.random.randint(low=0,high=int(2**16),size=self.n_ca)
		seeds = []
		for i in range(self.n_ca):
			seeds.append(self.get_next_seed())
		if s.parent is None: #first generation
			magnitudes = np.ones_like(seeds)
		else:
			self.set_params(s)
			param_values = self.policy.get_param_values(trainable=True)
			directions = []
			for seed in seeds:
				np.random.seed(seed)
				direction = np.random.normal(size=param_values.shape)
				directions.append(direction)
			magnitudes, constraint_vals = \
					self.optimizer.get_magnitudes(directions=directions,inputs=all_input_values,max_constraint_val=self.step_size)
		for (seed,magnitude) in zip(seeds,magnitudes):
			actions.append((seed,magnitude))
			# sp,r = self.getNextState(s,(seed,magnitude))
			# self.set_params(sp)
			# divergence = self.f_divergence(*all_input_values)
			# print("divergence: ",divergence)
		return actions

	@overrides
	def rollout(self, s):
		self.set_params(s)
		paths = self.obtain_samples(0)
		undiscounted_returns = [sum(path["rewards"]) for path in paths]
		if not (self.top_paths is None):
			action_seqs = [path["actions"] for path in paths]
			[self.top_paths.enqueue(action_seq,R,make_copy=True) for (action_seq,R) in zip(action_seqs,undiscounted_returns)]
		samples_data = self.process_samples(0, paths)
		# assert len(self.s[s].ca) == 0
		self.s[s].ca = self.getCandidateActions(s,samples_data)
		q = self.evaluate(samples_data)
		self.s[s].v = q
		self.record_tabular()
		return q
