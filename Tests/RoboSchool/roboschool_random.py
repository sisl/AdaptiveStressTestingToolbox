import roboschool
import gym
from garage.envs.normalized_env import normalize
from mylab.envs.tfenv import TfEnv
from mylab.envs.seed_env import SeedEnv
import numpy as np

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

os = []
o = env.reset()
os.append(o)
max_path_length = 100
path_length = 0
done = False
c_r = 0.0
actions = np.ones_like(env.action_space.sample())
while (path_length < max_path_length) and (not done):
	path_length += 1
	o, r, done, _ = env.step(actions)
	c_r += r
	os.append(o)
print('c_r: ',c_r)

os_ = []
o = env.reset()
os_.append(o)
max_path_length = 100
path_length = 0
done = False
c_r = 0.0
# actions = np.ones_like(env.action_space.sample())
while (path_length < max_path_length) and (not done):
	path_length += 1
	o, r, done, _ = env.step(actions)
	c_r += r
	os_.append(o)
print('c_r: ',c_r)

for (o,o_) in zip(os,os_):
	print(np.array_equal(o,o_))



