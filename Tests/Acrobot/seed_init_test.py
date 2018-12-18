import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

# from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from sandbox.rocky.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.misc import logger
from rllab.envs.normalized_env import normalize
from rllab.envs.env_spec import EnvSpec
from sandbox.rocky.tf.envs.base import to_tf_space

from Acrobot.acrobot import AcrobotEnv

from mylab.algos.gatrd import GATRD
from mylab.utils.np_weight_init import init_param_np

import os.path as osp
import argparse
# from example_save_trials import *
import tensorflow as tf
import joblib
import math
import numpy as np

seed = 1

np.random.seed(seed)
tf.set_random_seed(seed)
with tf.Session() as sess:
	# Create env
	env = TfEnv(AcrobotEnv(success_reward = 100,
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
	for param in params:
		print(param.name)
		print(tf.shape(param).eval())
		print(sess.run(param))
		init_param_np(param, policy, np.random)
		print(sess.run(param))

		

	