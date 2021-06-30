import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

from garage.tf.algos.trpo import TRPO
from garage.baselines.zero_baseline import ZeroBaseline
from traffic.make_env import make_env
from mylab.envs.tfenv import TfEnv
from garage.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy

from garage.misc import logger
import os.path as osp
import tensorflow as tf
import joblib
import time
import numpy as np

import pdb

with tf.Session() as sess:
    data = joblib.load("Data/Train/TRPO/seed0/itr_1000.pkl")
    agent = data['policy']
    env = data['env']

    max_path_length = 100

    obs = env.reset()
    agent.reset()
    path_length = 0
    cr = 0.
    env.render()
    while True:
        a, agent_info = agent.get_action(obs)
        a = np.argmax(agent_info["prob"])
        obs, reward, done, env_info = env.step(a)
        path_length += 1
        cr += reward
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
            print('return: ',cr)
            pdb.set_trace()
            obs = env.reset()
            agent.reset()
            path_length = 0
            cr = 0.
            env.render()
    # return