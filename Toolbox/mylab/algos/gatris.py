import rllab.misc.logger as logger
import tensorflow as tf
from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
import numpy as np


from mylab.algos.gatr import GATR

class GATRIS(GATR):
	"""
	Genetic Algorithm with Trust Region Mutation
	"""

	def __init__(
			self,
			**kwargs):
		super(GATRIS, self).__init__(**kwargs)


	@overrides
	def get_fitness(self, itr, all_paths):
		fitness = np.zeros(self.pop_size)
		for p in range(self.pop_size):
			self.set_params(itr,p)
			self.sum_other_weights[p] = 0.0
			for p_key in all_paths.keys():
				all_input_values = self.data2inputs(all_paths[p_key])
				path_lrs = self.f_path_lrs(*all_input_values)

				if not p_key == p:
					self.sum_other_weights[p] += np.sum(path_lrs)

				rewards = all_paths[p_key]["rewards"]
				valid_rewards = rewards*all_paths[p_key]["valids"]
				path_rewards = np.sum(valid_rewards,-1)
				fitness[p] += np.sum(path_lrs*path_rewards)
		return fitness

