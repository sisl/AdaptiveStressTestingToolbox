import roboschool
import gym
from garage.envs.normalized_env import normalize
from mylab.envs.tfenv import TfEnv
from mylab.envs.seed_env import SeedEnv
import numpy as np

env = TfEnv(normalize(gym.make('RoboschoolAnt-v1')))
env2 = TfEnv(normalize(SeedEnv(gym.make('RoboschoolAnt-v1'),random_reset=False,reset_seed=0)))

env.seed(0)
o01 = env.reset()
o02 = env2.reset()
print(np.array_equal(o01,o02))

o012 = env.reset()
o022 = env2.reset()
print(np.array_equal(o01,o012))
print(np.array_equal(o02,o022))

o12, r12, d12, _ = env2.step(np.ones_like(env.action_space.sample()))
env2.reset()
o122, r122, d122, _ = env2.step(np.ones_like(env.action_space.sample()))
print(np.array_equal(o12,o122))



