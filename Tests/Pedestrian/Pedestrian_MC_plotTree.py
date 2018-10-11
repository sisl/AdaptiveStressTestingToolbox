import mcts.AdaptiveStressTesting as AST
import mcts.ASTSim as ASTSim
import mcts.MCTSdpw as MCTSdpw
import mcts.AST_MCTS as AST_MCTS
import mcts.tree_plot as tree_plot
import numpy as np

from Pedestrian.av_simulator import AVSimulator
from Pedestrian.av_reward import AVReward
from Pedestrian.av_spaces import AVSpaces
from mylab.envs.ast_env import ASTEnv

import math

np.random.seed(0)

stress_test_num = 2
max_path_length = 20#50
ec = 100.0
n = 160
k=0.5
alpha=0.85
clear_nodes=False
top_k = 10

RNG_LENGTH = 2
SEED = 0 


reward_function = AVReward()
spaces = AVSpaces(interactive=True)
sim = AVSimulator(use_seed=True,spaces=spaces,max_path_length=max_path_length)


env = ASTEnv(interactive=True,
                             sample_init_state=False,
                             s_0=[-0.5, -4.0, 1.0, 11.17, -35.0],
                             simulator=sim,
                             reward_function=reward_function,
                             )

ast_params = AST.ASTParams(max_path_length,RNG_LENGTH,SEED)
ast = AST.AdaptiveStressTest(ast_params, env)

macts_params = MCTSdpw.DPWParams(max_path_length,ec,n,k,alpha,clear_nodes,top_k)
if stress_test_num == 2:
	result,tree = AST_MCTS.stress_test2(ast,macts_params,verbose=False, return_tree=True)
else:
	result,tree = AST_MCTS.stress_test(ast,macts_params,verbose=False, return_tree=True)
#reward, action_seq = result.rewards[1], result.action_seqs[1]
print("setp count: ",ast.step_count)

for (i,action_seq) in enumerate(result.action_seqs):
	reward, _ = ASTSim.play_sequence(ast,action_seq,sleeptime=0.0)
	print("predic reward: ",result.rewards[i])
	print("actual reward: ",reward)	

tree_plot.plot_tree(tree,d=max_path_length,path="Data/tree"+str(stress_test_num)+"_"+str(max_path_length),format="png")
