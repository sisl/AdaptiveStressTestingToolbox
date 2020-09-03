# import base class
# useful packages for math and debugging
import numpy as np

from ast_toolbox.rewards import ASTReward


# Define the class, inherit from the base
class ExampleATReward(ASTReward):
    def __init__(self):
        super().__init__()

    def give_reward(self, action, **kwargs):
        # get the info from the simulator
        info = kwargs['info']
        robustness = info["d"]
        is_goal = info["is_goal"]
        is_terminal = info["is_terminal"]

        # update reward and done bool

        if (is_goal):  # We found a crash
            # reward = -10*robustness
            reward = 0
        elif (is_terminal):
            # reward = 0
            # Heuristic reward based on distance between car and ped at end
            # heuristic_reward = dist
            reward = -1e5 - 1e4 * robustness  # We reached the horizon with no crash
            # reward = -robustness
        else:
            reward = -robustness  # No failure

        return reward