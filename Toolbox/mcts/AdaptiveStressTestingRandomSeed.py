import copy
from mcts.AdaptiveStressTesting import AdaptiveStressTest
import mcts.MDP as MDP
import mcts.MCTSdpw as MCTSdpw
import numpy as np
import mcts.RNGWrapper as RNG

class ASTParams:
	def __init__(self,max_steps,rsg_length,init_seed,log_interval,log_tabular):
		self.max_steps = max_steps
		self.rsg_length = rsg_length
		self.init_seed = init_seed
		self.log_interval = log_interval
		self.log_tabular = log_tabular
		
class AdaptiveStressTestRS(AdaptiveStressTest):
	def __init__(self,**kwargs):
		super(AdaptiveStressTestRS, self).__init__(**kwargs)
		self.rsg = RNG.RSGInit(self.params.rsg_length, self.params.init_seed)
		self.initial_rsg = copy.deepcopy(self.rsg)

	def reset_rsg1(self):
		self.rsg = copy.deepcopy(self.initial_rsg)
	def random_action(self):
		self.rsg.next1()
		return ASTAction(action=copy.deepcopy(self.rsg))
	def explore_action(self,s,tree):
		self.rsg.next1()
		return ASTAction(action=copy.deepcopy(self.rsg))

class ASTAction:
	def __init__(self,action):
		self.action = action
	def __hash__(self):
		return hash(self.action)
	def __eq__(self,other):
		return self.action == other.action
	def get(self):
		return self.action.state