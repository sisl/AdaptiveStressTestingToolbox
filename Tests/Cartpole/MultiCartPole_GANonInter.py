import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

# from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.misc import logger
from rllab.envs.normalized_env import normalize
from rllab.envs.env_spec import EnvSpec
from sandbox.rocky.tf.envs.base import to_tf_space

from mylab.rewards.ast_reward import ASTReward
from mylab.envs.ast_env import ASTEnv
from mylab.simulators.policy_simulator import PolicySimulator

from Cartpole.cartpole import CartPoleEnv

from mylab.algos.ga import GA

import os.path as osp
import argparse
# from example_save_trials import *
import tensorflow as tf
import joblib
import math
import numpy as np

import mcts.BoundedPriorityQueues as BPQ
import csv
# Logger Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="cartpole")
parser.add_argument('--n_trial', type=int, default=10)
parser.add_argument('--n_itr', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=4000)
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_dir', type=str, default='./Data/AST/GANonInter')


parser.add_argument('--tabular_log_file', type=str, default='tab.txt')
parser.add_argument('--text_log_file', type=str, default='tex.txt')
parser.add_argument('--params_log_file', type=str, default='args.txt')
parser.add_argument('--log_tabular_only', type=bool, default=False)
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
logger.push_prefix("[%s] " % args.exp_name)

top_k = 10
max_path_length = 100
interactive = False

tf.set_random_seed(0)
with open(osp.join(log_dir, 'cartpole_GANonInter.csv'), mode='w') as csv_file:
	fieldnames = ['step_count']
	for i in range(top_k):
		fieldnames.append('reward '+str(i))
	writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
	writer.writeheader()

	sess = tf.Session()
	sess.__enter__()

	for trial in range(args.n_trial):
		if trial == 0:
			reuse = False
		else:
			reuse = True
		np.random.seed(trial)
		with tf.variable_scope("ast",reuse=reuse):
			# Instantiate the env
			env_inner = CartPoleEnv(use_seed=False)
			data = joblib.load("Data/Train/itr_50.pkl")
			policy_inner = data['policy']
			reward_function = ASTReward()

			simulator = PolicySimulator(env=env_inner,policy=policy_inner,max_path_length=max_path_length)
			env = TfEnv(ASTEnv(interactive=interactive,
										 simulator=simulator,
										 sample_init_state=False,
										 s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
										 reward_function=reward_function,
										 ))

			# Create policy
			policy = GaussianLSTMPolicy(name='lstm_policy',
										env_spec=env.spec,
										hidden_dim=128,
										use_peepholes=True)
			params = policy.get_params()
			sess.run(tf.variables_initializer(params))

			# Instantiate the RLLAB objects
			baseline = LinearFeatureBaseline(env_spec=env.spec)
			top_paths = BPQ.BoundedPriorityQueueInit(top_k)
			algo = GA(
				env=env,
				policy=policy,
				baseline=baseline,
				batch_size=args.batch_size,
				pop_size = 100,
				elites = 20,
				keep_best = 3,
				step_size=0.01,
				n_itr=args.n_itr,
				store_paths=False,
				max_path_length=max_path_length,
				top_paths = top_paths,
				plot=False,
				)

			algo.train(sess=sess, init_var=False)

			row_content = dict()
			row_content['step_count'] = args.n_itr*args.batch_size
			i = 0
			for (r,action_seq) in algo.top_paths:
				row_content['reward '+str(i)] = r
				i += 1
			writer.writerow(row_content)