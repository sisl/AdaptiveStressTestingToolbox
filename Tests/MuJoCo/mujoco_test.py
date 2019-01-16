import gym
from garage.envs.normalized_env import normalize
from mylab.envs.tfenv import TfEnv
import numpy as np

env = TfEnv(normalize(gym.make('Humanoid-v2')))

env.seed(0)
o01 = env.reset()
env.seed(1)
o11, r11, d11, _ = env.step(np.ones_like(env.action_space.sample()))

env.seed(0)
o02 = env.reset()
env.seed(2)
o21, r21, d21, _ = env.step(np.ones_like(env.action_space.sample()))

print('reset: ',np.array_equal(o01,o02))
print('step: ',np.array_equal(o11,o21))
print(r11,r21,d11,d21)