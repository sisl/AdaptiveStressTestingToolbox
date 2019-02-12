import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

# from garage.tf.algos.trpo import TRPO
from garage.baselines.zero_baseline import ZeroBaseline
from mylab.envs.tfenv import TfEnv
from garage.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from garage.misc import logger

from Acrobot.acrobot import AcrobotEnv

from mylab.utils.tree_plot import plot_tree, plot_node_num
from mylab.algos.psmctstrc import PSMCTSTRC

from mylab.algos.ddpg import DDPG
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.replay_buffer import SimpleReplayBuffer

import os.path as osp
import argparse
# from example_save_trials import *
import tensorflow as tf
import joblib
import math
import numpy as np

import mcts.BoundedPriorityQueues as BPQ
import csv
# Log Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="cartpole")
parser.add_argument('--n_trial', type=int, default=5)
parser.add_argument('--trial_start', type=int, default=0)
parser.add_argument('--n_itr', type=int, default=2500)
parser.add_argument('--batch_size', type=int, default=8000)
parser.add_argument('--step_size', type=float, default=0.01)
parser.add_argument('--n_epoch_cycles', type=int, default=20)
parser.add_argument('--snapshot_mode', type=str, default="none")
parser.add_argument('--snapshot_gap', type=int, default=5000)
parser.add_argument('--log_dir', type=str, default='./Data/TRPO')
parser.add_argument('--args_data', type=str, default=None)
args = parser.parse_args()

top_k = 10
max_path_length = 400
interactive = True

tf.set_random_seed(0)
sess = tf.Session()
sess.__enter__()

# Instantiate the env
env = TfEnv(AcrobotEnv(max_path_length = max_path_length,
						success_threshhold = 1.9999,
						torque_noise_max = 0.1,
						initial_condition_max = 0.1))

# Create policy
policy = ContinuousMLPPolicy(
	name='ast_agent',
	env_spec=env.spec,
	hidden_sizes=(128, 64, 32),
	hidden_nonlinearity=tf.nn.relu,
	output_nonlinearity=tf.nn.tanh,
)

qf = ContinuousMLPQFunction(
	env_spec=env.spec,
	hidden_sizes=(128, 64, 32),
	hidden_nonlinearity=tf.nn.relu)

replay_buffer = SimpleReplayBuffer(
	env_spec=env.spec, size_in_transitions=int(1e6), time_horizon=100)

action_noise = OUStrategy(env.spec, sigma=0.2)

with open(osp.join(args.log_dir, 'total_result.csv'), mode='w') as csv_file:
	fieldnames = ['step_count']
	for i in range(top_k):
		fieldnames.append('reward '+str(i))
	writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
	writer.writeheader()

	for trial in range(args.trial_start,args.trial_start+args.n_trial):
		# Create the logger
		log_dir = args.log_dir+'/'+str(trial)

		tabular_log_file = osp.join(log_dir, 'process.csv')
		# text_log_file = osp.join(log_dir, 'text.txt')
		params_log_file = osp.join(log_dir, 'args.txt')

		logger.set_snapshot_dir(log_dir)
		logger.set_snapshot_mode(args.snapshot_mode)
		logger.set_snapshot_gap(args.snapshot_gap)
		logger.log_parameters_lite(params_log_file, args)
		if trial > args.trial_start:
			old_log_dir = args.log_dir+'/'+str(trial-1)
			logger.pop_prefix()
			# logger.remove_text_output(osp.join(old_log_dir, 'text.txt'))
			logger.remove_tabular_output(osp.join(old_log_dir, 'process.csv'))
		# logger.add_text_output(text_log_file)
		logger.add_tabular_output(tabular_log_file)
		logger.push_prefix("["+args.exp_name+'_trial '+str(trial)+"_1]")

		np.random.seed(trial)

		algo = DDPG(
			env,
			policy=policy2,
			policy_lr=1e-4,
			qf_lr=1e-3,
			qf=qf,
			replay_buffer=replay_buffer,
			plot=False,
			target_update_tau=1e-2,
			n_epochs=args.n_itr2,#500,
			n_epoch_cycles=args.n_epoch_cycles,#20,
			rollout_batch_size=args.batch_size/max_path_length,#1, #rollout_batch_size is actually n_envs
			max_path_length=max_path_length,
			n_train_steps=50,
			discount=0.9,
			min_buffer_size=int(1e4),
			exploration_strategy=action_noise,
			policy_optimizer=tf.train.AdamOptimizer,
			qf_optimizer=tf.train.AdamOptimizer,
			top_paths=top_paths,
		   )

		algo.train(sess=sess, init_var=True)

		row_content = dict()
		row_content['step_count'] = args.batch_size*args.n_itr*args.n_epoch_cycles
		i = 0
		for (r,action_seq) in algo.top_paths:
			row_content['reward '+str(i)] = r
			i += 1
		writer.writerow(row_content)