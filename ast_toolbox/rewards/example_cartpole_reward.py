# import base class
from ast_toolbox.rewards.ast_reward import ASTReward

# useful packages for math and debugging
import numpy as np
import pdb

# Define the class, inherit from the base
class CartpoleReward(ASTReward):
	# standard ast reward function from https://ieeexplore.ieee.org/abstract/document/7311450
	def __init__(self,
				const1 = 1e4,
				const2 = 1e3):
		self.const1 = const1
		self.const2 = const2
		super().__init__()


	def give_reward(self, action, **kwargs):
	    # get the info from the simulator
	    info = kwargs['info']
	    is_goal = info["is_goal"]
	    is_terminal = info["is_terminal"]
	    dist = info["dist"]
	    prob = info["prob"]
	    # ast_action_seq = info["ast_action_seq"]

	    # update reward and done bool
	    # reward = np.abs(np.mean(ast_action_seq))
	    reward = np.log(prob)
	    # reward = 0.0
	    if (is_goal): # We found a crash
	        reward += 0.0
	        # reward -= np.sum(np.abs(ast_action_seq))
	    elif (is_terminal):
	        reward += -self.const1 - self.const2 * dist # We reached the horizon with no crash
	    return reward