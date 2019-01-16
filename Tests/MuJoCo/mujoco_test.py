import gym
from garage.envs.normalized_env import normalize
from mylab.envs.tfenv import TfEnv

env = TfEnv(normalize(gym.make('Humanoid-v2')))

env.seed(0)
o = env.reset()