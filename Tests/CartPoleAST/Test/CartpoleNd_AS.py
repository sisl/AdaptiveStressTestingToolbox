import mcts.AdaptiveStressTesting as AST
import mcts.ASTSim as ASTSim
import mcts.MCTSdpw as MCTSdpw
import mcts.AST_MCTS as AST_MCTS
import mcts.BoundedPriorityQueues as BPQ
import numpy as np
from mylab.envs.ast_env import ASTEnv
from mylab.simulators.policy_simulator import PolicySimulator
from cartpole import CartPoleEnv
import tensorflow as tf
import math

from ast_reward_standard import ASTRewardS
from cartpole_simulator import CartpoleSimulator
# from ast_env import ASTEnv
from tfenv import TfEnv

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

import joblib

seed = 0
np.random.seed(0)
tf.set_random_seed(0)

max_path_length = 100
ec = 100.0
k = 0.5
alpha = 0.85
clear_nodes = False
M = 10
n = 2
top_k = 10
top_paths = BPQ.BoundedPriorityQueue(top_k)
stress_test_num = 1

with tf.Session() as sess:
	# Instantiate the policy
	env_inner = CartPoleEnv(nd=1,use_seed=False)
	data = joblib.load("control_policy.pkl")
	policy_inner = data['policy']
	reward_function = ASTRewardS()

	# Create the environment
	simulator = PolicySimulator(env=env_inner,policy=policy_inner,max_path_length=100)
	env = ASTEnv(interactive=True,
								 simulator=simulator,
	                             sample_init_state=False,
	                             s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
	                             reward_function=reward_function,
	                             )
	# simulator = CartpoleSimulator(sut=policy_inner,max_path_length=100,use_seed=False,nd=1)
	# env = ASTEnv(open_loop=False,
	# 							 simulator=simulator,
	# 							 fixed_init_state=True,
	# 							 s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
	# 							 reward_function=reward_function,
	# 							 )
	env = TfEnv(env)

	ast_params = AST.ASTParams(max_path_length,batch_size=1000,log_tabular=True)
	ast = AST.AdaptiveStressTest(ast_params, env, top_paths)

	macts_params = MCTSdpw.DPWParams(max_path_length,ec,n,k,alpha,clear_nodes,top_k)
	
	if stress_test_num == 2:
		result = AST_MCTS.stress_test2(ast,macts_params,top_paths,False)
	else:
		result = AST_MCTS.stress_test(ast,macts_params,top_paths,False)
	#reward, action_seq = result.rewards[1], result.action_seqs[1]
	print("setp count: ",ast.step_count)

	for (i,action_seq) in enumerate(result.action_seqs):
		reward, _ = ASTSim.play_sequence(ast,action_seq,sleeptime=0.0)
		print("predic reward: ",result.rewards[i])
		print("actual reward: ",reward)	

