import os
import os.path as osp
import time

import joblib
import numpy as np
import tensorflow as tf
from CartPole.cartpole import CartPoleEnv
from garage.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.misc import logger
from garage.tf.algos.trpo import TRPO
from garage.tf.envs.base import TfEnv
from garage.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy

os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU




env = TfEnv(CartPoleEnv(use_seed=False))

with tf.Session() as sess:
    data = joblib.load("../Cartpole/control_policy.pkl")
    agent = data['policy']

    o = env.reset()
    agent.reset()
    path_length = 0
    env.render()
    max_path_length = 100

    total_r = 0
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        a = np.argmax(agent_info["prob"])
        next_o, r, d, env_info = env.step(a)
        path_length += 1
        total_r += r
        if d:
            break
        o = next_o
        env.render()
        timestep = 0.05
        time.sleep(timestep)
        print("step: ",path_length," total reward: ",total_r)
    # return
