import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

import numpy as np
import tensorflow as tf
from garage.misc import logger
import math

from mylab.algos.mcts import MCTS

import os.path as osp
import argparse

import joblib
import mcts.BoundedPriorityQueues as BPQ
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="cartpole")
parser.add_argument('--itr', type=int, default=30000)
parser.add_argument('--k',type=float, default=0.5)
parser.add_argument('--alpha',type=float, default=0.5)
parser.add_argument('--ec',type=float, default=10.0)#1.414)
parser.add_argument('--log_interval', type=int, default=10000) # per step_num
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=50)
parser.add_argument('--log_dir', type=str, default='./Data/AST/MCTSAS')
parser.add_argument('--args_data', type=str, default=None)
args = parser.parse_args()

# Create the logger
log_dir = args.log_dir\
            +('K'+str(args.k))\
            +('A'+str(args.alpha))\
            +('Ec'+str(args.ec))\
            +('/'+'seed'+str(args.seed))
args.log_dir = log_dir

tabular_log_file = osp.join(log_dir, 'progress.csv')
text_log_file = osp.join(log_dir, 'text.txt')
params_log_file = osp.join(log_dir, 'args.txt')

logger.log_parameters_lite(params_log_file, args)
# logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode(args.snapshot_mode)
logger.set_snapshot_gap(args.snapshot_gap)
logger.set_log_tabular_only(False)
logger.push_prefix("[%s] " % args.exp_name)

seed = args.seed
top_k = 10

import mcts.BoundedPriorityQueues as BPQ
top_paths = BPQ.BoundedPriorityQueue(top_k)

np.random.seed(seed)
tf.set_random_seed(seed)
with tf.Session() as sess:

	# Instantiate the env
	from traffic.make_env import make_env
	env_inner = make_env(env_name='highway')
	data = joblib.load("Data/Train/TRPO/seed0/itr_1000.pkl")
	policy_inner = data['policy']

	from mylab.rewards.ast_reward import ASTReward
	from mylab.envs.ast_env import ASTEnv
	from mylab.simulators.policy_simulator import PolicySimulator
	from mylab.envs.tfenv import TfEnv
	reward_function = ASTReward()
	simulator = PolicySimulator(env=env_inner,policy=policy_inner,max_path_length=100)
	env = TfEnv(ASTEnv(interactive=True,
	                             simulator=simulator,
	                             sample_init_state=False,
	                             s_0=0., # not used
	                             reward_function=reward_function,
	                             ))

	algo = MCTS(
	        env=env,
	        max_path_length=100,
	        ec=args.ec,
	        n_itr=args.itr+1,
	        k=args.k,
	        alpha=args.alpha,
	        clear_nodes=True,
	        log_interval=args.log_interval,
	        top_paths=top_paths,
	        log_dir=args.log_dir,
	        gamma=1.0,
	        stress_test_mode=2,
	        log_tabular=True,
	        plot_tree=False,
	        plot_path=None,
	        plot_format='png'
			)

	algo.train(runner=None)



