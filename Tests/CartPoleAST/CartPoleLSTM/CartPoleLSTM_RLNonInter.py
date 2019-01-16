import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

# from garage.tf.algos.trpo import TRPO
from garage.baselines.zero_baseline import ZeroBaseline
from mylab.envs.tfenv import TfEnv
from garage.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.misc import logger
from garage.envs.normalized_env import normalize
from garage.envs.env_spec import EnvSpec


from mylab.rewards.ast_reward import ASTReward
from mylab.envs.ast_env import ASTEnv
from mylab.simulators.policy_simulator import PolicySimulator

from CartPoleAST.CartPole.cartpole import CartPoleEnv

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
parser.add_argument('--exp_name', type=str, default='cartpole_exp')
parser.add_argument('--tabular_log_file', type=str, default='tab.txt')
parser.add_argument('--text_log_file', type=str, default='tex.txt')
parser.add_argument('--params_log_file', type=str, default='args.txt')
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_tabular_only', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default='./Data/AST/RLNonInter')
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
	env_inner = CartPoleEnv(use_seed=False)
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
	reward_function = ASTReward()

	# Create the environment
	max_path_length = 100
	simulator = PolicySimulator(env=env_inner,policy=policy_inner,max_path_length=max_path_length)
	env = TfEnv(ASTEnv(interactive=False,
								 simulator=simulator,
	                             sample_init_state=False,
	                             s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
	                             reward_function=reward_function,
	                             ))

	# Instantiate the RLLAB objects
	baseline = ZeroBaseline(env_spec=env.spec)
	optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
	algo = TRPO(
	    env=env,
	    policy=policy,
	    baseline=baseline,
	    batch_size=4000,
	    step_size=0.1,
	    n_itr=1,#25,
	    store_paths=True,
	    optimizer= optimizer,
	    max_path_length=max_path_length,
	    plot=False,
	    )

	algo.train(sess=sess, init_var=False)