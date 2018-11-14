from mylab.algos.gais import GAIS
import numpy as np
from rllab.misc.overrides import overrides

class GAISN(GAIS):
	"""
	Genetic Algorithm with Importance Sampling, normalize IS weights
	"""

	def __init__(
			self,
			**kwargs):

		super(GAISN, self).__init__(**kwargs)

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

				log_path_lrs = np.sum(valid_log_lrs,-1)
				Log_path_lrs = np.append(Log_path_lrs,log_path_lrs)

				rewards = all_paths[p_key]["rewards"]
				valid_rewards = rewards*all_paths[p_key]["valids"]
				path_rewards = np.sum(valid_rewards,-1)
				Path_rewards = np.append(Path_rewards, path_rewards)
			lse_lrs = log_sum_exp(Log_path_lrs,0)
			importance_weights = np.exp(Log_path_lrs - lse_lrs)
			fitness[p] += np.sum(Path_rewards*importance_weights)
		return fitness

def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner
    """
    return log_sum_exp(x, dim) - np.log(x.shape[dim])


def log_sum_exp(x, dim):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner
    """
    max_x = np.max(x,dim)
    new_x = x - np.repeat(np.expand_dims(max_x,dim),x.shape[dim],dim)
    return max_x + np.log(np.sum(np.exp(new_x),dim))

