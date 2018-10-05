import mcts.AdaptiveStressTestingActionSpace as AST_AS
import mcts.ASTSim as ASTSim
import mcts.MCTSdpw as MCTSdpw
import mcts.AST_MCTS as AST_MCTS
import numpy as np

from Pedestrian.av_simulator import AVSimulator
from Pedestrian.av_reward import AVReward
from Pedestrian.av_spaces import AVSpaces
from mylab.envs.ast_env import ASTEnv

import math

np.random.seed(0)

max_path_length = 50
ec = 100.0
n = 160
top_k = 10

RNG_LENGTH = 2
SEED = 0 


reward_function = AVReward()
spaces = AVSpaces(interactive=True)
sim = AVSimulator(use_seed=False,spaces=spaces,max_path_length=max_path_length)


env = ASTEnv(interactive=True,
                             sample_init_state=False,
                             s_0=[-0.5, -4.0, 1.0, 11.17, -35.0],
                             simulator=sim,
                             reward_function=reward_function,
                             )

actions = [env.action_space.sample() for i in range(200)]
d = False
R = 0.0
step = 0
env.reset()
while not d:
	o,r,d,i = env.step(actions[step])
	R += r
	step += 1
print(step,R)

d = False
R = 0.0
step = 0
env.reset()
while not d:
	o,r,d,i = env.step(actions[step])
	R += r
	step += 1
print(step,R)

