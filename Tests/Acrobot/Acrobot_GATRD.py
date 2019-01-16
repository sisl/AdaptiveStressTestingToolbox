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

import os.path as osp
import argparse
# from example_save_trials import *
import tensorflow as tf
import joblib
import math
import numpy as np

# Logger Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='cartpole_exp')
parser.add_argument('--tabular_log_file', type=str, default='progress.csv')
parser.add_argument('--text_log_file', type=str, default='tex.txt')
parser.add_argument('--params_log_file', type=str, default='args.txt')
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_tabular_only', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default='./Data/AST/GATRD/Test')
parser.add_argument('--args_data', type=str, default=None)
args = parser.parse_args()

# Create the logger
log_dir = args.log_dir

tabular_log_file = osp.join(log_dir, args.tabular_log_file)
text_log_file = osp.join(log_dir, args.text_log_file)
params_log_file = osp.join(log_dir, args.params_log_file)

logger.log_parameters_lite(params_log_file, args)
logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
prev_snapshot_dir = logger.get_snapshot_dir()
prev_mode = logger.get_snapshot_mode()
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode(args.snapshot_mode)
logger.set_snapshot_gap(args.snapshot_gap)
logger.set_log_tabular_only(args.log_tabular_only)
logger.push_prefix("[%s] " % args.exp_name)

seed = 1
top_k = 10
max_path_length = 100

import mcts.BoundedPriorityQueues as BPQ
top_paths = BPQ.BoundedPriorityQueue(top_k)

np.random.seed(seed)
tf.set_random_seed(seed)
with tf.Session() as sess:
	# Create env
	env = TfEnv(AcrobotEnv(max_path_length = max_path_length,
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

	# Instantiate the RLLAB objects
	baseline = ZeroBaseline(env_spec=env.spec)
	# optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

	algo = GATRD(
		env=env,
		policy=policy,
		baseline=baseline,
		batch_size= max_path_length,
		pop_size = 5,
		truncation_size = 3,
		keep_best = 1,
		step_size = 1.0,
		n_itr = 2,
		store_paths=False,
		# optimizer= optimizer,
		max_path_length=max_path_length,
		top_paths=top_paths,
		plot=False,
		)

	algo.train(sess=sess, init_var=False)

	