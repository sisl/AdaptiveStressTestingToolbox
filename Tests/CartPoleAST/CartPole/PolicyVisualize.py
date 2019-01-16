import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

from garage.tf.algos.trpo import TRPO
from garage.baselines.zero_baseline import ZeroBaseline
from CartPoleAST.CartPole.cartpole import CartPoleEnv
from mylab.envs.tfenv import TfEnv
from garage.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy

from garage.misc import logger
import os.path as osp
import tensorflow as tf
import joblib
import time
import numpy as np

env = TfEnv(CartPoleEnv())

with tf.Session() as sess:
    data = joblib.load("Data/Train/itr_50.pkl")
    agent = data['policy']

    o = env.reset()
    agent.reset()
    path_length = 0
    env.render()
    max_path_length = 100

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        a = np.argmax(agent_info["prob"])
        next_o, r, d, env_info = env.step(a)
        path_length += 1
        if d:
            break
        o = next_o
        env.render()
        timestep = 0.05
        time.sleep(timestep)
        print(path_length)
    # return