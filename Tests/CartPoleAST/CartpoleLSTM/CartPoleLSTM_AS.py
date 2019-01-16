import mcts.AdaptiveStressTestingActionSpace as AST_AS
import mcts.ASTSim as ASTSim
import mcts.MCTSdpw as MCTSdpw
import mcts.AST_MCTS as AST_MCTS
import numpy as np
from mylab.rewards.ast_reward import ASTReward
from mylab.envs.ast_env import ASTEnv
from mylab.simulators.policy_simulator import PolicySimulator
from CartPoleAST.CartPole.cartpole import CartPoleEnv
import tensorflow as tf
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

import joblib

np.random.seed(0)

max_path_length = 100
ec = 100.0
n = 1
top_k = 10

with tf.Session() as sess:
	# Instantiate the policy
	env_inner = CartPoleEnv(use_seed=False)
	data = joblib.load("Data/Train/itr_50.pkl")
	policy_inner = data['policy']
	reward_function = ASTReward()

	# Create the environment
	simulator = PolicySimulator(env=env_inner,policy=policy_inner,max_path_length=max_path_length)
	env = ASTEnv(interactive=True,
								 simulator=simulator,
	                             sample_init_state=False,
	                             s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
	                             reward_function=reward_function,
	                             )

	ast_params = AST_AS.ASTParams(max_path_length)
	ast = AST_AS.AdaptiveStressTestAS(ast_params, env)

	#ASTSim.sample(ast)

	#macts_params = MCTSdpw.DPWParams(50,100.0,100,0.5,0.85,1.0,0.0,True,1.0e308,np.uint64(0),10)
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

