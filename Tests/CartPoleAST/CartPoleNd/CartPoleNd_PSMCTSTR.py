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


from mylab.rewards.ast_reward import ASTReward
from mylab.envs.ast_env import ASTEnv
from mylab.simulators.policy_simulator import PolicySimulator
from mylab.utils.tree_plot import plot_tree

from CartPoleNd.cartpole_nd import CartPoleNdEnv

from mylab.algos.psmctstr import PSMCTSTR

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
parser.add_argument('--log_dir', type=str, default='./Data/AST/PSMCTSTR/Test')
parser.add_argument('--args_data', type=str, default=None)
args = parser.parse_args()

# Create the logger
log_dir = args.log_dir

tabular_log_file = osp.join(log_dir, args.tabular_log_file)
text_log_file = osp.join(log_dir, args.text_log_file)
params_log_file = osp.join(log_dir, args.params_log_file)

logger.log_parameters_lite(params_log_file, args)
# logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
prev_snapshot_dir = logger.get_snapshot_dir()
prev_mode = logger.get_snapshot_mode()
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode(args.snapshot_mode)
logger.set_snapshot_gap(args.snapshot_gap)
logger.set_log_tabular_only(args.log_tabular_only)
logger.push_prefix("[%s] " % args.exp_name)

seed = 0
top_k = 10
max_path_length = 100

import mcts.BoundedPriorityQueues as BPQ
top_paths = BPQ.BoundedPriorityQueue(top_k)

np.random.seed(seed)
tf.set_random_seed(seed)
with tf.Session() as sess:
	# Create env
	env_inner = CartPoleNdEnv(nd=10,use_seed=False)
	data = joblib.load("../CartPole/Data/Train/itr_50.pkl")
	policy_inner = data['policy']
	reward_function = ASTReward()

	simulator = PolicySimulator(env=env_inner,policy=policy_inner,max_path_length=max_path_length)
	env = TfEnv(ASTEnv(interactive=True,
								 simulator=simulator,
								 sample_init_state=False,
								 s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
								 reward_function=reward_function,
								 ))

	# Create policy
	policy = DeterministicMLPPolicy(
		name='ast_agent',
		env_spec=env.spec,
		hidden_sizes=(64, 32)
	)

	params = policy.get_params()
	sess.run(tf.variables_initializer(params))

	# Instantiate the RLLAB objects
	baseline = ZeroBaseline(env_spec=env.spec)
	# optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

	algo = PSMCTSTR(
		env=env,
		policy=policy,
		baseline=baseline,
		batch_size=max_path_length,
		step_size=0.01,
		n_itr=10,
		initial_pop=1,
		max_path_length=max_path_length,
		top_paths=top_paths,
		seed=0,
		ec = 100.0,
		k=0.5,
		alpha=0.85,
		log_interval=1,
		plot=False,
		)

	algo.train(sess=sess, init_var=False)
	plot_tree(algo.s,d=max_path_length,path=log_dir+"/tree",format="png")

	