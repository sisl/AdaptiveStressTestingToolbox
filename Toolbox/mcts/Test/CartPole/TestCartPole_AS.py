import AdaptiveStressTestingActionSpace as AST_AS
import ASTSim
import MCTSdpw
import AST_MCTS
import numpy as np
from Cartpole.policy_simulator import PolicySimulator
from Cartpole.cartpole import CartPoleEnv
from Cartpole.ast_reward_wrapper import ASTRewardWrapper
from policy_env import PolicyEnv
import tensorflow as tf
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

import joblib

np.random.seed(0)

max_path_length = 100
ec = 100.0
n = 10
top_k = 10

with tf.Session() as sess:
	env_inner = CartPoleEnv()
	data = joblib.load("/home/maxiaoba/DRLAST/TestEnv/Cartpole/Data/Train/itr_50.pkl")
	policy_inner = data['policy']
	sim = PolicySimulator(env_inner, policy_inner, max_path_length=max_path_length)
	reward_function = ASTRewardWrapper()
	env = PolicyEnv(interactive=True,
	                             sample_init_state=False,
	                             s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
	                             simulator=sim,
	                             reward_function=reward_function,
	                             vectorized = True,
	                             # spaces=spaces
	                             )

	ast_params = AST_AS.ASTParams(max_path_length)
	ast = AST_AS.AdaptiveStressTestAS(ast_params, env)

	#ASTSim.sample(ast)

	#macts_params = MCTSdpw.DPWParams(50,100.0,100,0.5,0.85,1.0,0.0,True,1.0e308,np.uint64(0),10)
	macts_params = MCTSdpw.DPWParams(max_path_length,ec,n,0.5,0.85,1.0,0.0,True,1.0e308,np.uint64(0),top_k)
	stress_test_num = 1
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

