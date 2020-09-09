# Define the class, inherit from the base
class ASTReward(object):
    """Function to calculate the rewards for timesteps when optimizing AST solver policies.

    """

    def __init__(self):
        pass

    def give_reward(self, action, **kwargs):
        """Returns the reward for a given time step.

        Parameters
        ----------
        action : array_like
            Action taken by the AST solver.
        kwargs :
            Accepts relevant info for computing the reward.
        Returns
        -------
        reward : float
            Reward based on the previous action.
        """

        raise NotImplementedError
