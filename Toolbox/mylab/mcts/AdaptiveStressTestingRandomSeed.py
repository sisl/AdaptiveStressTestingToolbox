import copy
from mylab.mcts.AdaptiveStressTesting import AdaptiveStressTest
import mylab.mcts.MDP as MDP
import mylab.mcts.MCTSdpw as MCTSdpw
import numpy as np
import mylab.mcts.RNGWrapper as RNG
		
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
		action = self.action.state
		return action