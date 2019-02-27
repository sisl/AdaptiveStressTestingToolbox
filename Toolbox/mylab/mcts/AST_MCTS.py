import numpy as np
import mylab.mcts.MCTSdpw as MCTSdpw
import mylab.mcts.MDP as MDP

class StressTestResults:
	def __init__(self,rewards,action_seqs,q_values):
		self.rewards = rewards
		self.action_seqs = action_seqs
		self.q_values = q_values

def StressTestResultsInit(k):
	rewards = np.zeros(k)
	action_seqs = [None]*k
	q_values = [None]*k
	return StressTestResults(rewards,action_seqs,q_values)

def rollout_getAction(ast):
	def rollout_policy(s,tree):
		return ast.random_action()
	return rollout_policy

def explore_getAction(ast):
	def explore_policy(s,tree):
		return ast.explore_action(s,tree)
	return explore_policy

def stress_test(ast,mcts_params,top_paths,verbose=True,return_tree=False):
	# dpw_model = MCTSdpw.DPWModel(ast.transition_model,uniform_getAction(ast.rsg),uniform_getAction(ast.rsg))
	# dpw_model = MCTSdpw.DPWModel(ast.transition_model,uniform_getAction(ast),uniform_getAction(ast))
	dpw_model = MCTSdpw.DPWModel(ast.transition_model,rollout_getAction(ast),explore_getAction(ast))
	dpw = MCTSdpw.DPW(mcts_params,dpw_model,top_paths)
	(mcts_reward,action_seq) = MDP.simulate(dpw.f.model,dpw,MCTSdpw.selectAction,verbose=verbose)
	results = StressTestResultsInit(top_paths.N)
	#results = StressTestResultsInit(dpw.top_paths.length())
	#print(dpw.top_paths.length())

	# k = 0
	# for (r,tr) in dpw.top_paths:
	# 	results.rewards[k] = r
	# 	results.action_seqs[k] = tr.get_actions()
	# 	results.q_values[k] = tr.get_q_values()
	# 	k += 1

	# if mcts_reward >= results.rewards[0]:
	# 	print("mcts_reward = ",mcts_reward," top reward = ",results.rewards[0])
	results = ast.top_paths
	if return_tree:
		return results,dpw.s
	else:
		return results

def stress_test2(ast,mcts_params,top_paths,verbose=True,return_tree=False):
	mcts_params.clear_nodes = False
	# mcts_params.n *= ast.params.max_steps

	dpw_model = MCTSdpw.DPWModel(ast.transition_model,rollout_getAction(ast),explore_getAction(ast))
	dpw = MCTSdpw.DPW(mcts_params,dpw_model,top_paths)

	s = dpw.f.model.getInitialState()
	MCTSdpw.selectAction(dpw,s,verbose=verbose)
	# results = StressTestResultsInit(top_paths.N)
	# k = 0
	# for (r,tr) in dpw.top_paths:
	# 	results.rewards[k] = r
	# 	results.action_seqs[k] = tr.get_actions()
	# 	results.q_values[k] = tr.get_q_values()
	# 	k += 1
	results = ast.top_paths
	if return_tree:
		return results,dpw.s
	else:
		return results

