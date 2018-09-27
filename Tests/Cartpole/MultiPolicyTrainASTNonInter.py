import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

# from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from Cartpole.cartpole import CartPoleEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.misc import logger
from rllab.envs.normalized_env import normalize
from rllab.envs.env_spec import EnvSpec
from sandbox.rocky.tf.envs.base import to_tf_space

from policy_simulator import PolicySimulator
from ast_reward_wrapper import ASTRewardWrapper
from ast_spaces_wrapper import ASTSpacesWrapper
from policy_env import PolicyEnv

from Cartpole.cartpole import CartPoleEnv

from Pedestrian.ast_env import ASTEnv
from mylab.ast_vectorized_sampler import ASTVectorizedSampler
from mylab.ast_single_sampler import ASTSingleSampler
from mylab.algos.trpo import TRPO

import os.path as osp
import argparse
# from example_save_trials import *
import tensorflow as tf
import joblib
import math

import BoundedPriorityQueues as BPQ
import csv
# Logger Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="cartpole")
parser.add_argument('--n_trial', type=int, default=10)
parser.add_argument('--n_itr', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=4000)
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_dir', type=str, default='./Data/AST/NonInter')


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

with open(osp.join(log_dir, 'ast.csv'), mode='w') as csv_file:
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
		with tf.variable_scope("ast",reuse=reuse):
			# Instantiate the policy
			env_inner = CartPoleEnv()
			ast_spec = EnvSpec(
							observation_space=to_tf_space(env_inner.ast_observation_space),
							action_space=to_tf_space(env_inner.ast_action_space),
						)

			policy = GaussianLSTMPolicy(name='lstm_policy',
										env_spec=ast_spec,
										hidden_dim=128,
										use_peepholes=True)
			sess.run(tf.global_variables_initializer())

			# Instantiate the env
			data = joblib.load("Data/Train/itr_50.pkl")
			policy_inner = data['policy']
			sim = PolicySimulator(env_inner, policy_inner, max_path_length=100)
			reward_function = ASTRewardWrapper()
			# spaces = ASTSpacesWrapper(env_inner)

			# Create the environment
			# env = TfEnv(ASTEnv(action_only=False,
			env = TfEnv(PolicyEnv(interactive=False,
											sample_init_state=False,
											s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
											simulator=sim,
											reward_function=reward_function,
											vectorized = True,
											# spaces=spaces
											))

			# Instantiate the RLLAB objects
			baseline = LinearFeatureBaseline(env_spec=env.spec)
			optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
			# sampler_cls = ASTSingleSampler
			# sampler_cls = ASTVectorizedSampler
			top_paths = BPQ.BoundedPriorityQueueInit(top_k)
			algo = TRPO(
				env=env,
				policy=policy,
				baseline=baseline,
				batch_size=args.batch_size,
				step_size=0.1,
				n_itr=args.n_itr,
				store_paths=True,
				optimizer= optimizer,
				max_path_length=100,
				# sampler_cls=sampler_cls,
				# sampler_args={"sim": sim,
				#               "reward_function": reward_function,
				#               "interactive": True},
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