import numpy as np
import MCTSdpw
import MDP
import AdaptiveStressTesting as AST
import RNGWrapper as RNG

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

# def uniform_getAction(ast_rsg):
# 	if type(ast_rsg)==AST.AdaptiveStressTest:
# 		return uniform_getAction(ast_rsg.rsg)
# 	elif type(ast_rsg)==RNG.RSG:
# 		def policy(s,rng):
# 			return AST.random_action(ast_rsg)
# 		return policy

# def uniform_getAction(ast):
# 	def policy(s,rng):
# 		return ast.random_action()
# 	return policy

def rollout_getAction(ast):
	def rollout_policy(s,tree,rng):
		return ast.random_action()
	return rollout_policy

def explore_getAction(ast):
	def explore_policy(s,tree,rng):
		return ast.explore_action(s,tree)
	return explore_policy

def stress_test(ast,mcts_params,verbose=True):
	# dpw_model = MCTSdpw.DPWModel(ast.transition_model,uniform_getAction(ast.rsg),uniform_getAction(ast.rsg))
	# dpw_model = MCTSdpw.DPWModel(ast.transition_model,uniform_getAction(ast),uniform_getAction(ast))
	dpw_model = MCTSdpw.DPWModel(ast.transition_model,rollout_getAction(ast),explore_getAction(ast))
	dpw = MCTSdpw.DPWInit(mcts_params,dpw_model)
	(mcts_reward,action_seq) = MDP.simulate(dpw.f.model,dpw,MCTSdpw.selectAction,verbose=verbose)
	results = StressTestResultsInit(mcts_params.top_k)
	#results = StressTestResultsInit(dpw.top_paths.length())
	#print(dpw.top_paths.length())
	k = 0
	for (r,tr) in dpw.top_paths:
		results.rewards[k] = r
		results.action_seqs[k] = tr.get_actions()
		results.q_values[k] = tr.get_q_values()
		k += 1

	if mcts_reward >= results.rewards[0]:
		print("mcts_reward = ",mcts_reward," top reward = ",results.rewards[0])

	return results

def stress_test2(ast,mcts_params,verbose=True):
	mcts_params.clear_nodes = False
	mcts_params.n *= ast.params.max_steps

	dpw_model = MCTSdpw.DPWModel(ast.transition_model,rollout_getAction(ast),explore_getAction(ast))
	dpw = MCTSdpw.DPWInit(mcts_params,dpw_model)
	s = dpw.f.model.getInitialState(dpw.rng)
	MCTSdpw.selectAction(dpw,s,verbose=verbose)
	results = StressTestResultsInit(mcts_params.top_k)
	k = 0
	for (r,tr) in dpw.top_paths:
		results.rewards[k] = r
		results.action_seqs[k] = tr.get_actions()
		results.q_values[k] = tr.get_q_values()
		k += 1

	return results

