import gym
from garage.envs.normalized_env import normalize
from mylab.envs.tfenv import TfEnv
from mylab.envs.seed_env import SeedEnv
import numpy as np

from BipedalWalker.bipedalwalker import BipedalWalker
env = TfEnv(normalize(BipedalWalker()))

env.seed(0)
o01 = env.reset()
env.seed(1)
o11, r11, d11, _ = env.step(np.ones_like(env.action_space.sample()))

env.seed(1)
o02 = env.reset()

env.seed(0)
o01 = env.reset()
env.seed(2)
o12, r12, d12, _ = env.step(np.ones_like(env.action_space.sample()))

print('reset: ',np.array_equal(o01,o02))
print('step: ',np.array_equal(o11,o12))

env = TfEnv(normalize(SeedEnv(BipedalWalker(),random_reset=False,reset_seed=0)))
o01 = env.reset()
o11, r11, d11, _ = env.step(np.ones_like(env.action_space.sample()))
o02 = env.reset()
o12, r12, d12, _ = env.step(np.ones_like(env.action_space.sample()))
print('reset: ',np.array_equal(o01,o02))
print('step: ',np.array_equal(o11,o12))



