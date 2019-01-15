import garage.misc.logger as logger
import tensorflow as tf
from garage.misc import ext
from garage.misc.overrides import overrides
import garage.misc.logger as logger
from garage.tf.misc import tensor_utils
import tensorflow as tf
import numpy as np


from mylab.algos.gatr import GATR
from mylab.utils.exp_utils import log_sum_exp

class GATRISN(GATR):
	"""
	Genetic Algorithm with Trust Region Mutation and self-normalized Importance Sampling
	"""

	def __init__(
			self,
			**kwargs):
		super(GATRISN, self).__init__(**kwargs)


	@overrides
	def get_fitness(self, itr, all_paths):
		fitness = np.zeros(self.pop_size)
		for p in range(self.pop_size):
			# print("p: ",p)
			self.set_params(itr,p)
			self.sum_other_weights[p] = 0.0
			Path_rewards = np.array([])
			Log_path_lrs = np.array([])
			for p_key in all_paths.keys():
				all_input_values = self.data2inputs(all_paths[p_key])
				path_lrs = self.f_path_lrs(*all_input_values)
				if not p_key == p:
					self.sum_other_weights[p] += np.sum(path_lrs)

				log_path_lrs = np.log(path_lrs)

				rewards = all_paths[p_key]["rewards"]
				valid_rewards = rewards*all_paths[p_key]["valids"]
				path_rewards = np.sum(valid_rewards,-1)

				if self.f_F == "mean":
					Log_path_lrs = np.append(Log_path_lrs,log_path_lrs)
					Path_rewards = np.append(Path_rewards, path_rewards)
				else:
					max_indx = np.argmax(path_rewards)
					Log_path_lrs = np.append(Log_path_lrs,log_path_lrs[max_indx])
					Path_rewards = np.append(Path_rewards, path_rewards[max_indx])
					
			lse_lrs = log_sum_exp(Log_path_lrs,0)
			importance_weights = np.exp(Log_path_lrs - lse_lrs)
			fitness[p] += np.sum(Path_rewards*importance_weights)
		return fitness

