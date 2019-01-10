from garage.algos.base import RLAlgorithm
from garage.misc.overrides import overrides
import garage.misc.logger as logger
from garage.tf.algos.batch_polopt import BatchPolopt
from mylab.samplers.vectorized_ga_sampler import VectorizedGASampler
from mylab.utils.seeding import hash_seed
from mylab.utils.mcts_utils import *
import numpy as np
import tensorflow as tf

from mylab.utils import seeding

class MCTS(BatchPolopt):
	"""
	Policy Space MCTS
	"""
	def __init__(
			self,
			stress_test_num,
			n_itr,
			ec, #exploration constant
			k, #progress widening constant
			alpha, #progress widening constant
			top_paths, #BPQ
			# seed,
			max_path_length,
			log_interval = 4000,
			**kwargs):
		self.stress_test_num = stress_test_num
		self.n_itr = n_itr
		self.ec = ec 
		self.k = k
		self.alpha = alpha
		self.top_paths = top_paths
		self.log_interval = log_interval
		self.max_path_length = max_path_length
		self.s = {}
		# self.seed = 0
		self.stepNum = 0
		# self.np_random, _ = seeding.np_random(seed=self.seed) #used in set_params
		super(MCTS, self).__init__(**kwargs, sampler_cls=VectorizedGASampler)

	@overrides
	def init_opt(self):
		return dict()

	def getInitialState(self):
		self.t_index = 0
		self.env.reset()
		s0 = MCTSStateInit(self.t_index,None,None)
		self.sim_hash = s0.hash
		return s0

	def getNextAction(self,s):
		return self.env.action_space.sample()

	def getNextState(self,s,a):
		assert self.sim_hash == s0.hash
		self.t_index += 1
		o, r, done, info = self.env.step(a)
		self.step_count += 1
		if self.step_count%self.params.batch_size == 0:
			logger.record_tabular('StepNum',self.step_count)
			for (topi, path) in enumerate(self.top_paths):
				logger.record_tabular('reward '+str(topi), path[0])
			logger.dump_tabular(with_prefix=False)
		sp = MCTSStateInit(self.t_index, s, a)
		return sp, r

	@overrides
	def train(self):
		# self.initial()
		for i in range(self.n_itr):
			self.itr = i
			s0 = self.getInitialState()
			self.simulate(s0)

		self.shutdown_worker()
		if created_session:
			sess.close()

	def simulate(self, s, verbose=False):
		if not (s in self.s):
			self.s[s] = StateNode()
			return self.rollout(s)
		self.s[s].n += 1

		need_new_action = False
		if (s.parent is None) and (not (self.initial_pop == 0)):
			if len(self.s[s].a) < self.initial_pop:
				need_new_action = True
		elif len(self.s[s].a) < self.k*self.s[s].n**self.alpha:
			need_new_action = True

		if need_new_action:
			a = self.getNextAction(s)
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

		q = self.update(s,a,r,sp)
		return q

	def update(self, s, a, r, sp):
		q = r + self.simulate(sp)
		cA = self.s[s].a[a]
		cA.n += 1
		cA.q += (q-cA.q)/float(cA.n)
		self.s[s].a[a] = cA
		if self.f_Q == "mean":
			return q
		else:
			qs = [self.s[s].a[a_dummy].q for a_dummy in self.s[s].a.keys()]
			return np.max(qs)

	def rollout(self, s):
		self.set_params(s)
		paths = self.obtain_samples(0)
		undiscounted_returns = [sum(path["rewards"]) for path in paths]
		if not (self.top_paths is None):
			action_seqs = [path["actions"] for path in paths]
			[self.top_paths.enqueue(action_seq,R,make_copy=True) for (action_seq,R) in zip(action_seqs,undiscounted_returns)]
		samples_data = self.process_samples(0, paths)
		q = self.evaluate(samples_data)
		self.s[s].v = q
		self.record_tabular()
		return q

	@overrides
	def obtain_samples(self, itr):
		self.stepNum += self.batch_size
		return self.sampler.obtain_samples(itr)

	def record_tabular(self):
		if self.stepNum%self.log_interval == 0:
			logger.record_tabular('Itr',self.itr)
			logger.record_tabular('StepNum',self.stepNum)
			logger.record_tabular('TreeSize',len(self.s))
			if self.top_paths is not None:
				for (topi, path) in enumerate(self.top_paths):
					logger.record_tabular('reward '+str(topi), path[0])
			logger.dump_tabular(with_prefix=False)


class MCTSState:
	def __init__(self,t_index,hash,parent,action):
		self.t_index = t_index
		self.hash = hash
		self.parent = parent
		self.action = action
	def __hash__(self):
		if self.parent is None:
			return hash((self.t_index,None,hash(self.action)))
		else:
			return hash((self.t_index,self.parent.hash,hash(self.action)))
	def __eq__(self,other):
		return hash(self) == hash(other)

def MCTSStateInit(t_index,parent,action):
	obj = MCTSState(t_index,0,parent,action)
	obj.hash = hash(obj)
	return obj

def get_action_sequence(s):
	actions = []
	while not s.parent is None:
		actions.append(s.action)
		s = s.parent
	actions = list(reversed(actions))
	return actions