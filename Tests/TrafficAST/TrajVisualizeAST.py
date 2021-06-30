import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU
import os.path as osp
import argparse
import tensorflow as tf
import joblib
import math
import numpy as np
import time

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

from traffic.make_env import make_env

import pdb

with tf.Session() as sess:
    data = joblib.load("Data/AST/PSMCTSTRCK0.5A0.5Ec10.0CA4lr1.0FmeanQmax/seed2/itr_20000.pkl")
    env = data['env']
    top_paths = data['top_paths']

    max_path_length = 100

    for key in top_paths.keys():
        print('stored return: ',key)
        action_sequence = top_paths[key]

        obs = env.reset()
        path_length = 0
        cr = 0.
        for a in action_sequence:
            # print(a)
            obs, reward, done, env_info = env.step(a)
            path_length += 1
            cr += reward
            env.render()
            timestep = 0.05
            time.sleep(timestep)
            # print(path_length)
            # print('obs: ',obs)
            # print('reward: ',reward)
            # print('done: ',done)
            # print(env_info)

        print('actual return: ',cr)
        pdb.set_trace()