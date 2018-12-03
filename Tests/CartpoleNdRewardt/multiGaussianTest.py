from scipy.stats import multivariate_normal
import numpy as np
from mylab.algos.gais_n import log_mean_exp, log_sum_exp

for d in range(2,11):
	print("d: ",d)
	r = np.zeros(d)
	px = multivariate_normal.pdf(r, mean=0.0, cov=1.0)
	print(px)
	py = multivariate_normal.pdf(r, mean=1.0, cov=1.0)
	print(py)
	lr = np.exp(np.sum(np.log(py))-np.sum(np.log(px)))
	print(lr)
	
	# check log_sum_exp and log_mean_exp
	# array = np.zeros((2,d))
	# array[0,:] = px[:]
	# array[1,:] = py[:]

	# dim = 0
	# print("dim: ",dim)
	# lse = np.log(np.sum(np.exp(array),dim))
	# lse2 = log_sum_exp(array,dim)
	# print(lse)
	# print(lse2)

	# lme = np.log(np.mean(np.exp(array),dim))
	# lme2 = log_mean_exp(array,dim)
	# print(lme)
	# print(lme2)

	# dim = 1
	# print("dim: ",dim)
	# lse = np.log(np.sum(np.exp(array),dim))
	# lse2 = log_sum_exp(array,dim)
	# print(lse)
	# print(lse2)

	# lme = np.log(np.mean(np.exp(array),dim))
	# lme2 = log_mean_exp(array,dim)
	# print(lme)
	# print(lme2)