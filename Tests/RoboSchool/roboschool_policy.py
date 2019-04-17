import roboschool
import gym
from garage.envs.normalized_env import normalize
from mylab.envs.tfenv import TfEnv
from mylab.envs.seed_env import SeedEnv
import numpy as np
import time
import argparse
import joblib

data = joblib.load('Data/TRPOStep0.1/Pong-v1/0')
env.reset()
max_path_length = 100
path_length = 0
done = False
c_r = 0.0
while (path_length < max_path_length) and (not done):
	path_length += 1
	o, r, done, _ = env.step(np.ones_like(env.action_space.sample()))
	c_r += r
	env.render()
	print("step: ",path_length)
	print("o: ",o)
	print('r: ',r)
	print(done)
	time.sleep(0.1)
print('c_r: ',c_r)
