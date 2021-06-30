import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

from garage.tf.algos.trpo import TRPO
from garage.baselines.zero_baseline import ZeroBaseline
from traffic.make_env import make_env
from mylab.envs.tfenv import TfEnv
from garage.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy

from mylab.rewards.ast_reward import ASTReward
from mylab.envs.ast_env import ASTEnv
from mylab.simulators.policy_simulator import PolicySimulator

from garage.misc import logger
import os.path as osp
import tensorflow as tf
import joblib
import time
import numpy as np

import pdb

env = TfEnv(make_env(env_name='highway'))

with tf.Session() as sess:

    env_inner = make_env(env_name='highway')
    data = joblib.load("Data/Train/TRPO/seed0/itr_100.pkl")
    policy_inner = data['policy']
    reward_function = ASTReward()

    simulator = PolicySimulator(env=env_inner,policy=policy_inner,max_path_length=100)
    env = TfEnv(ASTEnv(interactive=True,
                                 simulator=simulator,
                                 sample_init_state=False,
                                 s_0=0., # not used
                                 reward_function=reward_function,
                                 ))

    max_path_length = 100

    obs = env.reset()
    path_length = 0
    # env.render()
    while True:
        action = np.zeros(env.action_space.low.size)
        for i in range(env.action_space.low.size):
            action[i] = float(input("Action[{}]\n".format(i)))

        obs, r, d, env_info = env.step(action)
        path_length += 1
        env.render()
        timestep = 0.05
        time.sleep(timestep)
        print(path_length)
        print('obs: ',obs)
        print('reward: ',reward)
        print('done: ',done)
        print(env_info)
        if done or (path_length > max_path_length):
            pdb.set_trace()
            obs = env.reset()
            path_length = 0
            # env.render()