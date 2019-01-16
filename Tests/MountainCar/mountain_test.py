import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

from garage.tf.algos.trpo import TRPO
from garage.baselines.zero_baseline import ZeroBaseline
from MountainCar.mountaincar import MountainCarEnv
from mylab.envs.tfenv import TfEnv
from garage.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy

from garage.misc import logger
import os.path as osp
import tensorflow as tf
import joblib
import time
import numpy as np

max_path_length = 100
env = TfEnv(MountainCarEnv(max_path_length = max_path_length))

with tf.Session() as sess:
    o = env.reset()
    path_length = 0
    total_reward = 0
    env.render()

    while path_length < max_path_length:
        if o[1] > 0.0:
            a = [0.95]
        else:
            a = [-0.95]

        next_o, r, d, env_info = env.step(a)
        path_length += 1
        total_reward += r
        if d:
            break
        o = next_o
        env.render()
        timestep = 0.05
        time.sleep(timestep)
        print(path_length)
        print(r)

    print('total_reward: ',total_reward)