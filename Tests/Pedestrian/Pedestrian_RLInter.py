import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

# from garage.tf.algos.trpo import TRPO
from garage.baselines.zero_baseline import ZeroBaseline
from CartPoleAST.CartPole.cartpole import CartPoleEnv
from mylab.envs.tfenv import TfEnv
from garage.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.misc import logger
from garage.envs.normalized_env import normalize
from garage.envs.env_spec import EnvSpec


from Pedestrian.av_simulator import AVSimulator
from Pedestrian.av_reward import AVReward
from Pedestrian.av_spaces import AVSpaces
from mylab.envs.ast_env import ASTEnv

from mylab.algos.trpo import TRPO

import os.path as osp
import argparse
# from example_save_trials import *
import tensorflow as tf
import joblib
import math
import numpy as np

# Logger Params
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='pedestrian_exp')
parser.add_argument('--tabular_log_file', type=str, default='tab.txt')
parser.add_argument('--text_log_file', type=str, default='tex.txt')
parser.add_argument('--params_log_file', type=str, default='args.txt')
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_tabular_only', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default='./Data/RLInter')
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

seed = 0
with tf.Session() as sess:
	np.random.seed(seed)
	tf.set_random_seed(seed)
	# Instantiate the policy
	np.random.seed(0)

	max_path_length = 50
	ec = 100.0
	n = 160
	top_k = 10

	RNG_LENGTH = 2


	reward_function = AVReward()
	spaces = AVSpaces(interactive=True)
	sim = AVSimulator(use_seed=False,spaces=spaces,max_path_length=max_path_length)


	env = TfEnv(ASTEnv(interactive=True,
	                             sample_init_state=False,
	                             s_0=[-0.5, -4.0, 1.0, 11.17, -35.0],
	                             simulator=sim,
	                             reward_function=reward_function,
	                             ))

	policy = GaussianLSTMPolicy(name='lstm_policy',
	                            env_spec=env.spec,
	                            hidden_dim=128,
	                            use_peepholes=True)
	sess.run(tf.global_variables_initializer())
	# Instantiate the RLLAB objects
	baseline = ZeroBaseline(env_spec=env.spec)
	optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
	# sampler_cls = ASTSingleSampler
	# sampler_cls = ASTVectorizedSampler
	algo = TRPO(
	    env=env,
	    policy=policy,
	    baseline=baseline,
	    batch_size=4000,
	    step_size=0.1,
	    n_itr=101,
	    store_paths=True,
	    optimizer= optimizer,
	    max_path_length=max_path_length,
	    # sampler_cls=sampler_cls,
	    # sampler_args={"sim": sim,
	    #               "reward_function": reward_function,
	    #               "interactive": True},
	    plot=False,
	    )

	algo.train(sess=sess, init_var=False)