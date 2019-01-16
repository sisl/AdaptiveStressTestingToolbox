import gym
from garage.envs.normalized_env import normalize
from mylab.envs.tfenv import TfEnv
from mylab.envs.seed_reset_env import SeedResetEnv
import numpy as np

env = TfEnv(normalize(gym.make('Humanoid-v2')))

# env.seed(0)
# o01 = env.reset()
# env.seed(1)
# o11, r11, d11, _ = env.step(np.ones_like(env.action_space.sample()))

# env.seed(1)
# o02 = env.reset()

# env.seed(0)
# o01 = env.reset()
# env.seed(2)
# o12, r12, d12, _ = env.step(np.ones_like(env.action_space.sample()))

# print('reset: ',np.array_equal(o01,o02))
# print('step: ',np.array_equal(o11,o12))

env2 = TfEnv(normalize(SeedResetEnv(gym.make('Humanoid-v2'),random_reset=False,reset_seed=0)))

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



