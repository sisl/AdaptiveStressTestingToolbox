import mcts.AdaptiveStressTesting as AST
import mcts.ASTSim as ASTSim
import mcts.MCTSdpw as MCTSdpw
import mcts.AST_MCTS as AST_MCTS
import numpy as np
from mylab.rewards.ast_reward import ASTReward
from mylab.envs.ast_env import ASTEnv
from mylab.simulators.policy_simulator import PolicySimulator
from Cartpole.cartpole import CartPoleEnv
import tensorflow as tf
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #just use CPU

import joblib
import csv

np.random.seed(0)

top_k = 10
n_trial= 5
interactive = False

stress_test_num = 2
max_path_length = 100
ec = 100.0
n = 1000
k=0.5
alpha=0.85
clear_nodes=True
top_k = 10
RNG_LENGTH = 2
SEED = 0

tf.set_random_seed(0)
with tf.Session() as sess:
	# Instantiate the policy
	env_inner = CartPoleEnv(use_seed=True)
	data = joblib.load("Data/Train/itr_50.pkl")
	policy_inner = data['policy']
	reward_function = ASTReward()

	simulator = PolicySimulator(env=env_inner,policy=policy_inner,max_path_length=max_path_length)
	env = ASTEnv(interactive=interactive,
								 simulator=simulator,
								 sample_init_state=False,
								 s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
								 reward_function=reward_function,
								 )

	with open('Data/AST/carploe_MCTS.csv', mode='w') as csv_file:
		fieldnames = ['step_count']
		for i in range(top_k):
			fieldnames.append('reward '+str(i))
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()

		for trial in range(n_trial):
			print("trial: ",trial)
			np.random.seed(trial)
			SEED = trial
			ast_params = AST.ASTParams(max_path_length,RNG_LENGTH,SEED)
			ast = AST.AdaptiveStressTest(ast_params, env)

			macts_params = MCTSdpw.DPWParams(max_path_length,ec,n,k,alpha,clear_nodes,top_k)
			stress_test_num = 2
			if stress_test_num == 2:
				result = AST_MCTS.stress_test2(ast,macts_params,verbose=False, return_tree=False)
			else:
				result = AST_MCTS.stress_test(ast,macts_params,verbose=False, return_tree=False)

			print("setp count: ",ast.step_count)
			print("rewards: ",result.rewards)

			# for (i,action_seq) in enumerate(result.action_seqs):
			# 	reward, _ = ASTSim.play_sequence(ast,action_seq,sleeptime=0.0)
			# 	print("predic reward: ",result.rewards[i])
			# 	print("actual reward: ",reward)	
			# 	print(ast.sim.log)

			row_content = dict()
			row_content['step_count'] = ast.step_count
			for j in range(top_k):
				row_content['reward '+str(j)] = result.rewards[j]
			writer.writerow(row_content)

