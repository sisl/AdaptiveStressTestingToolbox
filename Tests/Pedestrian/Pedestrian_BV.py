import mcts.AdaptiveStressTestingBlindValue as AST_BV
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

ast_params = AST_BV.ASTParams(max_path_length,ec)
ast = AST_BV.AdaptiveStressTestBV(p=ast_params, env=env)

macts_params = MCTSdpw.DPWParams(max_path_length,ec,n,0.5,0.85,1.0,0.0,True,1.0e308,np.uint64(0),top_k)
stress_test_num = 2
if stress_test_num == 2:
	result = AST_MCTS.stress_test2(ast,macts_params,False)
else:
	result = AST_MCTS.stress_test(ast,macts_params,False)
#reward, action_seq = result.rewards[1], result.action_seqs[1]
print("setp count: ",ast.step_count)

for (i,action_seq) in enumerate(result.action_seqs):
	reward, _ = ASTSim.play_sequence(ast,action_seq,sleeptime=0.0)
	print("predic reward: ",result.rewards[i])
	print("actual reward: ",reward)	

