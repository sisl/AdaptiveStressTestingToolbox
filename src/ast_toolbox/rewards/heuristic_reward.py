# useful packages for math and debugging
import numpy as np

# import base class
from ast_toolbox.rewards import ASTReward


# Define the class, inherit from the base
class HeuristicReward(ASTReward):
    def __init__(self,
                 action_model,  # the action model to get the log prob from
                 terminal_heuristic_coef=None  # numpy array of coefficients for each heuristic
                 ):

        self.action_model = action_model
        self.terminal_heuristic_coef = terminal_heuristic_coef
        if self.terminal_heuristic_coef is None:
            self.terminal_heuristic_coef = -10000 * np.ones(1)

        super().__init__()

    def give_reward(self, action, **kwargs):
        # get the info from the simulator
        info = kwargs['info']
        is_goal = info["is_goal"]
        is_terminal = info["is_terminal"]
        terminal_heuristics = info.get("terminal_heuristic", np.zeros(0))
        terminal_heuristics = np.append(np.ones(1), terminal_heuristics)
        # pdb.set_trace()
        if (is_goal):  # We found a crash
            reward = 0
        elif (is_terminal):
            reward = np.dot(self.terminal_heuristic_coef, terminal_heuristics)  # We reached the horizon with no crash
        else:
            reward = self.action_model.log_prob(action, **kwargs)  # No crash or horizon yet

        if np.isnan(reward):
            print("found nan")
        return reward
