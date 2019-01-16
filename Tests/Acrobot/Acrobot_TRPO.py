import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

# from garage.tf.algos.trpo import TRPO
from garage.baselines.linear_feature_baseline import LinearFeatureBaseline
from mylab.envs.tfenv import TfEnv
from garage.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from garage.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from garage.misc import logger
from garage.envs.normalized_env import normalize
from garage.envs.env_spec import EnvSpec


from Acrobot.acrobot import AcrobotEnv

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
parser.add_argument('--exp_name', type=str, default='HalfCheetah-v1')
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=1000)
parser.add_argument('--log_dir', type=str, default='./Data/AST/TRPO/Test')
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
max_path_length = 100

np.random.seed(seed)
tf.set_random_seed(seed)
with tf.Session() as sess:
    # Create env
    env = TfEnv(AcrobotEnv(max_path_length = max_path_length,
                            success_threshhold = 1.9999,
                            torque_noise_max = 0.0,))
    # print(env.observation_space)
    # print(env.action_space)
    # Create policy
    policy = GaussianMLPPolicy(
        name='ast_agent',
        env_spec=env.spec,
        hidden_sizes=(128, 64, 32)
    )
    # policy = GaussianLSTMPolicy(name='lstm_policy',
    #                             env_spec=ast_spec,
    #                             hidden_dim=128,
    #                             use_peepholes=True)
    
    params = policy.get_params()
    sess.run(tf.variables_initializer(params))

    # Instantiate the RLLAB objects
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    # optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        step_size=0.1,
        n_itr=25,
        store_paths=True,
        # optimizer= optimizer,
        max_path_length=max_path_length,
        plot=False,
        )

    algo.train(sess=sess, init_var=False)

    