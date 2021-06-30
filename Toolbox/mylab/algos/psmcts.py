import time
from garage.algos.base import RLAlgorithm
from garage.misc.overrides import overrides
import garage.misc.logger as logger
from garage.tf.algos.batch_polopt import BatchPolopt
from mylab.samplers.vectorized_sampler import VectorizedSampler
from mylab.utils.seeding import hash_seed
from mylab.utils.mcts_utils import *
import numpy as np
import tensorflow as tf

from mylab.utils import seeding
from mylab.utils.np_weight_init import init_policy_np

class PSMCTS(BatchPolopt):
	"""
	Policy Space MCTS
	"""
	def __init__(
			self,
			ec, #exploration constant
			k, #progress widening constant
			alpha, #progress widening constant
			top_paths, #BPQ
			f_F = "mean",
			f_Q = "max",
			step_size = 0.01,
			step_size_anneal = 1.0,
			log_interval = 4000,
			initial_pop = 0,
			**kwargs):
		self.ec = ec 
		self.k = k
		self.alpha = alpha
		self.top_paths = top_paths

		self.best_mean = -np.inf
		self.best_var = 0.0
		self.best_return = -np.inf
		self.best_s_mean = None
		self.best_s_max = None

		self.f_F = f_F
		self.f_Q = f_Q
		self.step_size = step_size
		self.step_size_anneal = 1.0
		self.log_interval = log_interval
		self.s = {}
		self.initial_pop = initial_pop
		self.step_num = 0
		self.np_random, seed = seeding.np_random() #used in set_params
		super(PSMCTS, self).__init__(**kwargs, sampler_cls=VectorizedSampler)
		self.policy.set_param_values(self.policy.get_param_values())

	@overrides
	def init_opt(self):
		return dict()

	def getInitialState(self):
		self.t_index = 0
		s0 = MCTSStateInit(self.t_index,None,None)
		return s0

	def getNextAction(self,s):
		seed = np.random.randint(low= 0, high = int(2**16))
		magnitude = self.step_size
		return (seed,magnitude)

	def getNextState(self,s,a):
		# assert self.t_index == s.t_index
		self.t_index += 1
		sp = MCTSStateInit(self.t_index, s, a)
		r = self.getReward(s,a,sp)
		return sp, r

	def getReward(self,s,a,sp):
		return 0.0

	def set_params(self,s):
		actions = get_action_sequence(s)
		param_values = np.zeros_like(self.policy.get_param_values(trainable=True))
		for i,(seed,magnitude) in enumerate(actions):
			self.np_random.seed(int(seed))
			if i==0 :
				param_values = init_policy_np(self.policy, self.np_random)
			else:
				param_values = param_values + magnitude*self.np_random.normal(size=param_values.shape)
		self.policy.set_param_values(param_values, trainable=True)

	def evaluate(self,undiscounted_returns):
		if self.f_F == "max":
			q = np.max(undiscounted_returns)
		else:
			q = np.mean(undiscounted_returns)
		return q

	@overrides
	def train(self, sess=None, init_var=False):
		created_session = True if (sess is None) else False
		if sess is None:
			sess = tf.Session()
			sess.__enter__()
		if init_var:
			sess.run(tf.global_variables_initializer())
		self.start_worker(sess)
		# self.initial()
		self.start_time = time.time()
		for i in range(self.n_itr):
			# print(self.s.keys())
			self.itr_start_time = time.time()
			self.itr = i
			s0 = self.getInitialState()
			self.simulate(s0)
			self.step_size = self.step_size*self.step_size_anneal

			logger.log("Saving snapshot...")
			params = self.get_itr_snapshot(i)
			if self.top_paths is not None:
				top_paths = dict()
				for (topi, path) in enumerate(self.top_paths):
					top_paths[path[1]] = path[0]
				params['top_paths'] = top_paths
			logger.save_itr_params(i, params)
			logger.log("Saved")

			self.record_tabular()

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
		if self.f_Q == "mean":
			cA.q += (q-cA.q)/float(cA.n)
			self.s[s].a[a] = cA
			return q
		else:
			if q > cA.q:
				cA.q = q
			self.s[s].a[a] = cA
			return cA.q

	def rollout(self, s):
		self.set_params(s)
		paths = self.obtain_samples(0)

		undiscounted_returns = [sum(path["rewards"]) for path in paths]
		if np.mean(undiscounted_returns) > self.best_mean:
			self.best_mean = np.mean(undiscounted_returns)
			self.best_var = np.var(undiscounted_returns)
			self.best_s_mean = s
		if np.max(undiscounted_returns) > self.best_return:
			self.best_return = np.max(undiscounted_returns)
			self.best_s_max = s
		if not (self.top_paths is None):
			action_seqs = [path["actions"] for path in paths]
			[self.top_paths.enqueue(action_seq,R,make_copy=True) for (action_seq,R) in zip(action_seqs,undiscounted_returns)]

		samples_data = self.process_samples(0, paths)
		q = self.evaluate(undiscounted_returns)
		self.s[s].v = q
		
		return q

	@overrides
	def obtain_samples(self, itr):
		self.step_num += self.batch_size
		paths = self.sampler.obtain_samples(itr)
		return paths

	def record_tabular(self):
		del logger._tabular[:]
		if self.itr%self.log_interval == 0:
			logger.record_tabular('Itr',self.itr)
			logger.record_tabular('Time', time.time() - self.start_time)
			logger.record_tabular('ItrTime', time.time() - self.itr_start_time)
			logger.record_tabular('StepNum',self.step_num)
			logger.record_tabular('TreeSize',len(self.s))
			if self.top_paths is not None:
				for (topi, path) in enumerate(self.top_paths):
					logger.record_tabular('reward '+str(topi), path[1])
			logger.record_tabular('BestMean', self.best_mean)
			logger.record_tabular('BestVar', self.best_var)
			logger.dump_tabular(with_prefix=False)

	@overrides
	def get_itr_snapshot(self, itr):
		self.set_params(self.best_s_mean)
		return dict(
			itr=itr,
			policy=self.policy,
			env=self.env,
		)


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