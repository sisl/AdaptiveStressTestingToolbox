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

	actions = [env.action_space.sample() for i in range(200)]
	d = False
	R = 0.0
	step = 0
	env.reset()
	while not d:
		o,r,d,i = env.step(actions[step])
		R += r
		step += 1
	print(step,R)

	d = False
	R = 0.0
	step = 0
	env.reset()
	while not d:
		o,r,d,i = env.step(actions[step])
		R += r
		step += 1
	print(step,R)





