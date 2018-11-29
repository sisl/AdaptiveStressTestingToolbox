from mylab.algos.psmctstr import PSMCTSTR
from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from mylab.samplers.vectorized_ga_sampler import VectorizedGASampler
from mylab.utils.seeding import hash_seed
from mylab.utils.mcts_utils import *
import numpy as np
import tensorflow as tf

class PSMCTSTRMQ(PSMCTSTR):
	"""
	Policy Space MCTS with max Q as action evaluaton
	"""
	def __init__(
			self,
			**kwargs):
		super(PSMCTSTRCMQ, self).__init__(**kwargs)

	@overrides
	def update(self, s, a, r, sp):
		q = r + self.simulate(sp)
		cA = self.s[s].a[a]
		cA.n += 1
		if q > cA.q:
			cA.q = q
		self.s[s].a[a] = cA
		return cA.q