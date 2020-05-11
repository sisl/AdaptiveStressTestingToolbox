# import base class
from ast_toolbox.rewards.ast_reward import ASTReward

# useful packages for math and debugging
import numpy as np
import pdb

# Define the class, inherit from the base
class ASTRewardT(ASTReward):
	#ast reward depends on trajectory probability
	def __init__(self,
				const1 = 1.0,
				const2 = 100):
		self.const1 = const1
		self.const2 = const2
		super().__init__()

	def give_reward(self, action, **kwargs):
	    # get the info from the simulator
	    info = kwargs['info']
	    is_goal = info["is_goal"]
	    is_terminal = info["is_terminal"]
	    dist = info["dist"]
	    assert dist > 0.0 and dist < 1.0
	    log_trajectory_pdf= info["log_trajectory_pdf"]
	    trajectory_pdf = np.exp(log_trajectory_pdf/self.const2)
	    if (is_goal): # We found a crash
	        reward = -0.5*self.const1/(self.const1+trajectory_pdf)+1.0
	    elif (is_terminal): # no crash founded and time out
	        reward = 0.5*(1-dist)
	    else:
	        reward = 0.0
	    return reward