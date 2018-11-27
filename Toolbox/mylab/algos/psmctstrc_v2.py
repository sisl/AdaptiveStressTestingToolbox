from mylab.algos.psmctstr import PSMCTSTR
from mylab.optimizers.direction_constraint_optimizer import DirectionConstraintOptimizer
from mylab.utils.mcts_utils import *
from rllab.misc.overrides import overrides
from rllab.misc import ext
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
import numpy as np


class StateNodeC:
	def __init__(self, a, ca, n, v):
		self.a = a #Dict{Action,StateActionNode}
		self.ca = ca #list of actions
		self.n = n #UInt64
		self.v = v
	def __init__(self):
		self.a = {}
		self.ca = []
		self.n = 0
		self.v = 0.0

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
	def simulate(self, s, verbose=False):
		if not (s in self.s):
			self.s[s] = StateNodeC()
			return self.rollout(s)
		self.s[s].n += 1
		if len(self.s[s].a) < self.k*self.s[s].n**self.alpha:
			if len(self.s[s].ca) == 0:
				self.s[s].ca = self.getNextActions(s)
			a = self.s[s].ca.pop()
			if not (a in self.s[s].a):
				self.s[s].a[a] = StateActionNode()
		else:
			cS = self.s[s]
			A = list(cS.a.keys())
			nA = len(A)
			UCT = np.zeros(nA)
			nS = cS.n
			for i in range(nA):
				cA = cS.a[A[i]]
				assert nS > 0
				assert cA.n > 0
				UCT[i] = cA.q + self.ec*np.sqrt(np.log(nS)/float(cA.n))
			a = A[np.argmax(UCT)]

		sp,r = self.getNextState(s,a)
		# print("new sp: ",sp in dpw.s.keys())
		if not (sp in self.s[s].a[a].s):
			self.s[s].a[a].s[sp] = StateActionStateNode()
			self.s[s].a[a].s[sp].r = r
			self.s[s].a[a].s[sp].n = 1
		else:
			self.s[s].a[a].s[sp].n += 1


		q = r + self.simulate(sp)
		cA = self.s[s].a[a]
		cA.n += 1
		cA.q += (q-cA.q)/float(cA.n)
		self.s[s].a[a] = cA
		return q

	def getNextActions(self,s,samples_data=None):
		actions = []
		self.set_params(s)
		if samples_data is None:
			paths = self.obtain_samples(0)
			samples_data = self.process_samples(0, paths)
		all_input_values = self.data2inputs(samples_data)
		seeds = np.random.randint(low=0,high=int(2**16),size=self.n_ca)
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
		actions = []
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
		assert len(self.s[s].ca) == 0
		self.s[s].ca = self.getNextActions(s,samples_data)
		q = self.evaluate(samples_data)
		self.s[s].v = q
		self.record_tabular()
		return q
