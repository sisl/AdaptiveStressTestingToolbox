import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from Cartpole.cartpole import CartPoleEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy

from rllab.misc import logger
import os.path as osp
import tensorflow as tf
import joblib
import time

with tf.Session() as sess:
    data = joblib.load("Data/AST/itr_100.pkl")
    agent = data['policy']
    env = data['env']
    o = env.reset()
    agent.reset()
    path_length = 0
    env.render()
    max_path_length = 100

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
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