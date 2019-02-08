# Import the example classes
from mylab.simulators.example_av_simulator import ExampleAVSimulator
from mylab.rewards.example_av_reward import ExampleAVReward
from mylab.spaces.example_av_spaces import ExampleAVSpaces

# Import the AST classes
from mylab.envs.ast_env import ASTEnv
from mylab.samplers.ast_vectorized_sampler import ASTVectorizedSampler
from mylab.algos.mcts import MCTS

# Import the necessary garage classes
from garage.tf.algos.trpo import TRPO
from garage.tf.envs.base import TfEnv
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.envs.normalized_env import normalize
from garage.misc import logger

# Useful imports
import os.path as osp
import argparse
from example_save_trials import *
import tensorflow as tf

# Logger Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='crosswalk_exp')
parser.add_argument('--tabular_log_file', type=str, default='tab.txt')
parser.add_argument('--text_log_file', type=str, default='tex.txt')
parser.add_argument('--params_log_file', type=str, default='args.txt')
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_tabular_only', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default='.')
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

# Instantiate the example classes
sim = ExampleAVSimulator()
reward_function = ExampleAVReward()
spaces = ExampleAVSpaces()

# Create the environment

seed = 0
top_k = 10

import mylab.mcts.BoundedPriorityQueues as BPQ
top_paths = BPQ.BoundedPriorityQueue(top_k)

env = normalize(ASTEnv(action_only=True,
                             fixed_init_state=True,
                             s_0=[-0.5, -4.0, 1.0, 11.17, -35.0],
                             simulator=sim,
                             reward_function=reward_function,
                             spaces=spaces
                             ))
algo = MCTS(
	    env=env,
		stress_test_num=2,
		max_path_length=100,
		ec=100.0,
		n_itr=1,
		k=0.5,
		alpha=0.85,
		clear_nodes=True,
		log_interval=1000,
	    top_paths=top_paths,
	    plot_tree=True,
	    plot_path=args.log_dir+'/tree'
	    )

algo.train()


