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
    data = joblib.load("Data/AST/TRPO/Test/itr_10.pkl")
    agent = data['policy']
    env = data['env']

    max_path_length = 101

    obs = env.reset()
    agent.reset()
    path_length = 0
    while True:
        a, agent_info = agent.get_action(obs)
        obs, reward, done, env_info = env.step(a)
        path_length += 1
        env.render()
        timestep = 0.05
        time.sleep(timestep)
        print(path_length)
        print('obs: ',obs)
        print('action: ',a)
        print('reward: ',reward)
        print('done: ',done)
        print(env_info)
        if done or (path_length > max_path_length):
            pdb.set_trace()
            obs = env.reset()
            agent.reset()
            path_length = 0
    # return