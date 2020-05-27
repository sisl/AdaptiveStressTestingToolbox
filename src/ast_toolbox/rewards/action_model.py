# useful packages for math and debugging


# Define the class, inherit from the base
class ActionModel(object):
    def __init__(self):
        pass

    def log_prob(self, action, **kwargs):
        """
        returns the log probability of an action
        Input
        -----
        action : the action to get the log probabilty of
        Outputs
        -------
        logp [Float] : log probability of the action
        """

        raise NotImplementedError
