import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

# from garage.tf.algos.trpo import TRPO
from garage.baselines.zero_baseline import ZeroBaseline
from mylab.envs.tfenv import TfEnv
from garage.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.misc import logger
from garage.envs.normalized_env import normalize

from mylab.rewards.ast_reward import ASTReward
from mylab.envs.ast_env import ASTEnv
from mylab.simulators.policy_simulator import PolicySimulator

from CartPoleAST.CartPole.cartpole import CartPoleEnv

from mylab.algos.trpo import TRPO

import os.path as osp
import argparse
# from example_save_trials import *
import tensorflow as tf
import joblib
import math
import numpy as np

seed = 0

np.random.seed(seed)
tf.set_random_seed(seed)
with tf.Session() as sess:
	env_inner = CartPoleEnv(use_seed=False)
	policy_inner = None
	reward_function = ASTReward()

	simulator = PolicySimulator(env=env_inner,policy=policy_inner,max_path_length=100)
	ast_env = ASTEnv(interactive=True,
								 simulator=simulator,
	                             sample_init_state=False,
	                             s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
	                             reward_function=reward_function,
	                             )
	print('ast_env: ',ast_env.vectorized)
	normal_env = normalize(ast_env)
	# print('normal_env: ',normal_env.vectorized)
	tf_env = TfEnv(normal_env)
	print('tf_env: ',tf_env.vectorized)

	