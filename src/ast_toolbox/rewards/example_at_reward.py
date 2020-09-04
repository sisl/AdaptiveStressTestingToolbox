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

        if (is_goal):
            reward = 0  # Found failure
        elif (is_terminal):
            reward = -1e5 - 1e4 * robustness
        else:
            reward = -robustness  # No failure

        return reward
