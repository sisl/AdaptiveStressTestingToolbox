import garage.misc.logger as logger
import tensorflow as tf
from garage.misc import ext
from garage.misc.overrides import overrides
import garage.misc.logger as logger
from garage.tf.misc import tensor_utils
import tensorflow as tf
import numpy as np

from mylab.algos.ga import GA
from mylab.utils.exp_utils import softmax

class GASM(GA):
	"""
	Genetic Algorithm with softmax parents selection, the score is the rand of the fitness
	"""

	def __init__(
			self,
			**kwargs):
		super(GASM, self).__init__(**kwargs)

	@overrides
	def select_parents(self, fitness):
		sort_indx = np.flip(np.argsort(fitness),axis=0)
		self.parents[0:self.keep_best] = sort_indx[0:self.keep_best]
		rank = np.zeros_like(fitness)
		rank[sort_indx] = np.flip(np.arange(len(fitness)),axis=0)
		prob = softmax(rank,0)
		self.parents[self.keep_best:self.pop_size] = np.random.choice(prob,size=self.pop_size-self.keep_best)
		# print("fitness: ",fitness)
		# print("sort_indx: ",sort_indx)
		# print("rank: ",rank)
		# print("prob: ",prob)
		# print("parents", self.parents)
