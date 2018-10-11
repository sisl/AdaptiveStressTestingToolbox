# import base class
from mylab.rewards.base import RewardFn

# useful packages for math and debugging
import numpy as np
import pdb

# Define the class, inherit from the base
class ASTReward(RewardFn):
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
	    # update reward and done bool
	    reward = np.log(1+prob)
	    if (is_goal): # We found a crash
	        reward += 0.0
	    elif (is_terminal):
	        reward += -self.const1 - self.const2 * dist # We reached
	        # the horizon with no crash
	    # else:
	        # reward = np.log(1+prob) # No crash or horizon yet
	    return reward