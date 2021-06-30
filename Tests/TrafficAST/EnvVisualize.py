import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

import time
import numpy as np

from traffic.make_env import make_env
from mylab.envs.tfenv import TfEnv

import pdb

env = TfEnv(make_env(env_name='highway',
                    driver_sigma=2.,
                    x_des_sigma=2.,
                    v0_sigma=0.,))


max_path_length = 100

obs = env.reset()
path_length = 0
env.render()
while True:
    action = input("Action in {}\n".format(env.rl_actions))
    action = int(action)
    while action < 0:
        t = 0
        cr = 0.
        env.reset()
        env.render()
        action = input("Action\n")
        action = int(action)
    obs, reward, done, env_info = env.step(action)
    path_length += 1
    env.render()
    time.sleep(0.05)
    print(path_length)
    print('obs: ',obs)
    print('reward: ',reward)
    print('done: ',done)
    print(env_info)
    if done or (path_length > max_path_length):
        pdb.set_trace()
        obs = env.reset()
        path_length = 0
        env.render()