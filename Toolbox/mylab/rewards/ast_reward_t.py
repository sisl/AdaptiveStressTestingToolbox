# import base class
from mylab.rewards.base import RewardFn

# useful packages for math and debugging
import numpy as np
import pdb

# Define the class, inherit from the base
class ASTReward(RewardFn):
	def __init__(self,
				const1 = 1):
		self.const1 = const1
		super().__init__()

	def give_reward(self, action, **kwargs):
	    # get the info from the simulator
	    info = kwargs['info']
	    is_goal = info["is_goal"]
	    is_terminal = info["is_terminal"]
	    dist = info["dist"]
	    trajectory_pdf= info["trajectory_pdf"]
	    if (is_goal): # We found a crash
	        reward = -0.5*self.const1/(self.const1+trajectory_pdf)+1.0
	    elif (is_terminal): # no crash founded and time out
	        reward = 0.5*(1-dist)
	    else:
	        reward = 0.0
	    return reward