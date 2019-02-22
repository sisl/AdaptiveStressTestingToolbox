import mcts.AdaptiveStressTestingBlindValue as AST_BV
import mcts.ASTSim as ASTSim
import mcts.MCTSdpw as MCTSdpw
import mcts.AST_MCTS as AST_MCTS
import mcts.BoundedPriorityQueues as BPQ
import numpy as np
from mylab.rewards.ast_reward import ASTReward
from mylab.envs.ast_env import ASTEnv
from mylab.simulators.policy_simulator import PolicySimulator
from CartPoleAST.CartPoleNd.cartpole_nd import CartPoleNdEnv
import tensorflow as tf
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

import joblib

np.random.seed(0)

max_path_length = 100
ec = 100.0
M = 10
n = 10
top_k = 10
top_paths = BPQ.BoundedPriorityQueue(top_k)

with tf.Session() as sess:
	# Instantiate the policy
	env_inner = CartPoleNdEnv(nd=5,use_seed=False)
	data = joblib.load("../CartPole/Data/Train/itr_50.pkl")
	policy_inner = data['policy']
	reward_function = ASTReward()

	# Create the environment
	simulator = PolicySimulator(env=env_inner,policy=policy_inner,max_path_length=100)
	env = ASTEnv(interactive=True,
								 simulator=simulator,
	                             sample_init_state=False,
	                             s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
	                             reward_function=reward_function,
	                             )

	ast_params = AST_BV.ASTParams(max_path_length,ec,M,batch_size=4000,log_tabular=True)
	ast = AST_BV.AdaptiveStressTest(ast_params, env, top_paths)

	#ASTSim.sample(ast)

	#macts_params = MCTSdpw.DPWParams(50,100.0,100,0.5,0.85,1.0,0.0,True,1.0e308,np.uint64(0),10)
	macts_params = MCTSdpw.DPWParams(max_path_length,ec,n,0.5,0.85,True,top_k)
	stress_test_num = 2
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

