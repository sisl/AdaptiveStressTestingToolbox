"""An example implementation of an ASTReward for an AV validation scenario."""
import numpy as np  # useful packages for math

from ast_toolbox.rewards import ASTReward  # import base class


# Define the class, inherit from the base
class CrazyTrolleyReward(ASTReward):
    """An example implementation of an ASTReward for an AV validation scenario.

    Parameters
    ----------
    num_peds : int
        The number of pedestrians in the scenario.
    cov_x : float
        Covariance of the x-acceleration.
    cov_y : float
        Covariance of the y-acceleration.
    cov_sensor_noise : float
        Covariance of the sensor noise.
    use_heuristic : bool
        Whether to include a heuristic in the reward based on how close the pedestrian is to the vehicle
        at the end of the trajectory.
    """

    def __init__(self,use_heuristic=False):
        self.use_heuristic = False
        super().__init__()

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
        # get the info from the simulator
        info = kwargs['info']
        frame_probability = info["frame_probability"]
        is_goal = info["is_goal"]
        is_terminal = info["is_terminal"]

        # update reward and done bool

        if (is_goal):  # We found a crash
            reward = 0
            print('########### is_goal ##############')
        elif (is_terminal):
            # reward = 0
            # Heuristic reward based on distance between car and ped at end
            if self.use_heuristic:
                heuristic_reward = 0
            else:
                # No Herusitic
                heuristic_reward = 0
            print('########### is_terminal ##############')
            reward = -100000 - 10000 * heuristic_reward  # We reached
            # the horizon with no crash
        else:
            print('########### else ##############')
            reward = np.log(frame_probability) # No crash or horizon yet

        return reward
