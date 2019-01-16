import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

# from garage.tf.algos.trpo import TRPO
from garage.baselines.zero_baseline import ZeroBaseline
from mylab.envs.tfenv import TfEnv
from garage.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.misc import logger
from garage.envs.normalized_env import normalize
from garage.envs.env_spec import EnvSpec


from Acrobot.acrobot import AcrobotEnv

from mylab.algos.gatrd import GATRD
from mylab.utils.np_weight_init import init_param_np, init_policy_np

import os.path as osp
import argparse
# from example_save_trials import *
import tensorflow as tf
import joblib
import math
import numpy as np

import time

seed = 1

np.random.seed(seed)
tf.set_random_seed(seed)
with tf.Session() as sess:
	# Create env
	env = TfEnv(AcrobotEnv(max_path_length = 100,
							success_threshhold = 1.9999,
							torque_noise_max = 0.0,))

	# Create policy
	policy = DeterministicMLPPolicy(
		name='ast_agent',
		env_spec=env.spec,
		hidden_sizes=(128, 64, 32)
	)

	params = policy.get_params()
	sess.run(tf.variables_initializer(params))
	param_values = policy.get_param_values(trainable=True)
	policy.set_param_values(param_values, trainable=True)

	# for param in params:
	# 	print(param.name)
	# 	print(tf.shape(param).eval())
	# 	print(sess.run(param))

	start_time = time.time()
	for param in params:
		init_param_np(param, policy, np.random)
	param_values = policy.get_param_values(trainable=True)
	print('Time1: ', time.time() - start_time)

	start_time = time.time()
	param_values = 0.3*np.random.normal(size=param_values.shape)
	print('Time2: ', time.time() - start_time)

	start_time = time.time()
	param_values = init_policy_np(policy, np_random=np.random)
	print('Time3: ', time.time() - start_time)

	policy.set_param_values(param_values, trainable=True)
	# for param in params:
	# 	print(param.name)
	# 	print(tf.shape(param).eval())
	# 	print(sess.run(param))


		

	