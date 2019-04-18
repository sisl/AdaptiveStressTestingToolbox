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

import roboschool
import gym
from garage.envs.normalized_env import normalize
from mylab.envs.tfenv import TfEnv
from mylab.envs.seed_env import SeedEnv

from mylab.algos.gatrd import GATRD

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
from mylab.utils.ga_argparser import get_ga_parser
args = get_ga_parser(log_dir='GATRD')
args.log_dir = './Data/'+args.exp_name+'/'+args.log_dir

top_k = 10
max_path_length = 500
interactive = True

tf.set_random_seed(0)
sess = tf.Session()
sess.__enter__()

# Instantiate the env
from RoboSchool.walkers import *
if args.exp_name == 'Reacher':
	from RoboSchool.reacher import RoboschoolReacher
	env = TfEnv(normalize(SeedEnv(RoboschoolReacher(),random_reset=False,reset_seed=0)))
elif args.exp_name == 'Hopper':
	env = TfEnv(normalize(SeedEnv(RoboschoolHopper(),random_reset=False,reset_seed=0)))
elif args.exp_name == 'Walker2d':
	env = TfEnv(normalize(SeedEnv(RoboschoolWalker2d(),random_reset=False,reset_seed=0)))
elif args.exp_name == 'HalfCheetah':
	env = TfEnv(normalize(SeedEnv(RoboschoolHalfCheetah(),random_reset=False,reset_seed=0)))
elif args.exp_name == 'Ant':
	env = TfEnv(normalize(SeedEnv(RoboschoolAnt(),random_reset=False,reset_seed=0)))
elif args.exp_name == 'Humanoid':
	env = TfEnv(normalize(SeedEnv(RoboschoolHumanoid(),random_reset=False,reset_seed=0)))
else:
	env = TfEnv(normalize(SeedEnv(gym.make('Roboschool'+args.exp_name),random_reset=False,reset_seed=0)))

# Create policy
policy = DeterministicMLPPolicy(
	name='ast_agent',
	env_spec=env.spec,
    hidden_sizes=(128, 64, 32),
    output_nonlinearity=tf.nn.tanh,
)

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
		logger.push_prefix("["+args.exp_name+'_trial '+str(trial)+"]")

		np.random.seed(trial)

		params = policy.get_params()
		sess.run(tf.variables_initializer(params))

		# Instantiate the RLLAB objects
		baseline = ZeroBaseline(env_spec=env.spec)
		top_paths = BPQ.BoundedPriorityQueue(top_k)
		algo = GATRD(
			env=env,
			policy=policy,
			baseline=baseline,
			batch_size=args.batch_size,
			pop_size=args.pop_size,
			truncation_size=args.truncation_size,
			keep_best=args.keep_best,
			step_size=args.step_size,
			step_size_anneal=args.step_size_anneal,
			n_itr=args.n_itr,
			store_paths=False,
			max_path_length=max_path_length,
			top_paths = top_paths,
			f_F=args.f_F,
			log_interval=args.log_interval,
			plot=False,
			)

		algo.train(sess=sess, init_var=False)

		row_content = dict()
		row_content['step_count'] = algo.stepNum
		i = 0
		for (r,action_seq) in algo.top_paths:
			row_content['reward '+str(i)] = r
			i += 1
		writer.writerow(row_content)