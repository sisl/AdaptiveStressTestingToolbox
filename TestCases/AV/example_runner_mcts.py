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
parser.add_argument('--run_num', type=int, default=0)

parser.add_argument('--iters', type=int, default=101)
parser.add_argument('--batch_size', type=int, default=4000)
parser.add_argument('--clip_range', type=float, default=0.3)
# Policy Params
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--policy', type=str, default="LSTM")
parser.add_argument('--use_peepholes', type=bool, default=False)
parser.add_argument('--load_policy', type=bool, default=False)

# Env Params
parser.add_argument('--action_only', type=bool, default=True)
parser.add_argument('--fixed_init_state', type=bool, default=False)

# MCTS Params
parser.add_argument('--alpha', type=float, default=0.85)
parser.add_argument('--k', type=float, default=0.5)
parser.add_argument('--ec', type=float, default=100.0)

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
y = [-2.125,-4.625]
x = [-0.5, 0.5]
vp = [0.5, 1.5]
vc = [9.755, 12.315]
xc = [-30.625, -39.375]

s_0 = [ x[np.mod(args.run_num,2)],
        y[np.mod(args.run_num//2, 2)],
        vp[np.mod(args.run_num//4, 2)],
        vc[np.mod(args.run_num//8, 2)],
        xc[np.mod(args.run_num//16, 2)]]
print(s_0)
s_0=[-0.0, -2.0, 1.0, 11.17, -35.0]
env = ASTEnv(action_only=True,
                             fixed_init_state=True,
                             s_0=s_0,
                             simulator=sim,
                             reward_function=reward_function,
                             spaces=spaces
                             )
algo = MCTS(
	    env=env,
		stress_test_num=1,
		max_path_length=50,
		ec=args.ec,
		n_itr=int(args.iters*args.batch_size/100**2),
		k=args.k,
		alpha=args.alpha,
		clear_nodes=True,
		log_interval=1000,
	    top_paths=top_paths,
	    plot_tree=False,
	    plot_path=args.log_dir+'/tree'
	    )

algo.train()
# n = np.zeros((50,6))
# for i in range(50):
# 	n[i,:] = spaces.action_space.sample()
#
# print(np.mean(n, axis=0))

