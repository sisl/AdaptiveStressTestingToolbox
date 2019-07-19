import roboschool
import gym
from garage.envs.normalized_env import normalize
from mylab.envs.tfenv import TfEnv
from mylab.envs.seed_env import SeedEnv
import numpy as np
import time
import argparse
import joblib
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='Ant')
parser.add_argument('--trial', type=int, default=0)
args = parser.parse_args()

np.random.seed(args.trial)
with tf.Session() as sess:
	data_path = './Data/Lexington/Ant/PSMCTSTRCK0.3A0.3Ec10.0Step1.0FmeanQmax/'+str(args.trial)+'/itr_50000.pkl'
	data = joblib.load(data_path)
	env = data['env']
	o = env.reset()
	env.render()

	policy = data['policy']
	max_path_length = 500
	path_length = 0
	done = False
	c_r = 0.0
	while (path_length < max_path_length) and (not done):
		path_length += 1
		a, _ = policy.get_action(o)
		o, r, done, _ = env.step(a)
		c_r += r
		env.render()
		print("step: ",path_length)
		print("o: ",o)
		print('r: ',r)
		print(done)
		time.sleep(0.1)
	print('c_r: ',c_r)