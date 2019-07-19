import roboschool
import gym
from garage.envs.normalized_env import normalize
from mylab.envs.tfenv import TfEnv
from mylab.envs.seed_env import SeedEnv
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='Ant-v1')
args = parser.parse_args()

from RoboSchool.walkers import *
if args.exp_name == 'Reacher':
	from RoboSchool.reacher import RoboschoolReacher
	env = TfEnv(normalize(SeedEnv(RoboschoolReacher(),random_reset=False,reset_seed=0)))
elif args.exp_name == 'Hopper':
	env = TfEnv(normalize(SeedEnv(RoboschoolHopper(),random_reset=False,reset_seed=0)))
elif args.exp_name == 'Walker2d':
	env = TfEnv(normalize(SeedEnv(RoboschoolWalker2d(),random_reset=False,reset_seed=0)))
elif args.exp_name == 'HalfCheetah':
	env = TfEnv(normalize(SeedEnv(RoboschoolHalfCheetah(),random_reset=False,reset_seed=0)))
elif args.exp_name == 'Ant':
	env = TfEnv(normalize(SeedEnv(RoboschoolAnt(),random_reset=False,reset_seed=0)))
elif args.exp_name == 'Humanoid':
	env = TfEnv(normalize(SeedEnv(RoboschoolHumanoid(),random_reset=False,reset_seed=0)))
else:
	env = TfEnv(normalize(SeedEnv(gym.make('Roboschool'+args.exp_name),random_reset=False,reset_seed=0)))

env.reset()
max_path_length = 100
path_length = 0
done = False
c_r = 0.0
while (path_length < max_path_length) and (not done):
	path_length += 1
	o, r, done, _ = env.step(env.action_space.sample())
	c_r += r
	env.render()
	import pdb
	pdb.set_trace()
	print("step: ",path_length)
	print("o: ",o)
	print('r: ',r)
	print(done)
	time.sleep(0.1)
print('c_r: ',c_r)
