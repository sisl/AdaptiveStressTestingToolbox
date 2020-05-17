import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

# from garage.tf.algos.trpo import TRPO
from src.ast_toolbox import TfEnv
from garage.misc import logger

from src.ast_toolbox.rewards import ASTRewardS
from src.ast_toolbox import ASTEnv
from CartPole.cartpole_simulator import CartpoleSimulator

from src.ast_toolbox.algos.mcts import MCTS

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
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_dir', type=str, default='./Data/AST/MCTSAS/Test')
parser.add_argument('--args_data', type=str, default=None)
args = parser.parse_args()

# Create the logger
log_dir = args.log_dir

tabular_log_file = osp.join(log_dir, 'process.csv')
text_log_file = osp.join(log_dir, 'text.csv')
params_log_file = osp.join(log_dir, 'args.txt')

logger.log_parameters_lite(params_log_file, args)
logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
prev_snapshot_dir = logger.get_snapshot_dir()
prev_mode = logger.get_snapshot_mode()
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode(args.snapshot_mode)
logger.set_snapshot_gap(args.snapshot_gap)
logger.set_log_tabular_only(False)
logger.push_prefix("[%s] " % args.exp_name)

seed = 0
top_k = 10

import src.ast_toolbox.mcts.BoundedPriorityQueues as BPQ
top_paths = BPQ.BoundedPriorityQueue(top_k)

np.random.seed(seed)
tf.set_random_seed(seed)
with tf.Session() as sess:
	# Create env
	# control_policy_path = "../CartPole/Data/Train/itr_5.pkl"
	control_policy_path = "../CartPole/control_policy.pkl"
	print(control_policy_path)
	data = joblib.load(control_policy_path)
	sut = data['policy']
	reward_function = ASTRewardS()

	simulator = CartpoleSimulator(sut=sut,max_path_length=100,use_seed=False,nd=1)
	env = ASTEnv(open_loop=False,
								 simulator=simulator,
								 fixed_init_state=True,
								 s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
								 reward_function=reward_function,
								 )
	env = TfEnv(env)

	algo = MCTS(
	    env=env,
		stress_test_num=2,
		max_path_length=100,
		ec=10.0,
		n_itr=100,
		k=0.5,
		alpha=0.5,
		clear_nodes=False,
		log_interval=1000,
	    top_paths=top_paths,
	    plot_tree=True,
	    plot_path=args.log_dir+'/tree'
	    )

	algo.train()

	