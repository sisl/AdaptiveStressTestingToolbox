"""An example implementation of an ASTReward for an AV validation scenario."""
import numpy as np  # useful packages for math

from ast_toolbox.rewards import ASTReward  # import base class


# Define the class, inherit from the base
class ExampleAVReward(ASTReward):
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

    def __init__(self,
                 num_peds=1,
                 cov_x=0.1,
                 cov_y=0.01,
                 cov_sensor_noise=0.1,
                 use_heuristic=True):

        self.c_num_peds = num_peds
        self.c_cov_x = cov_x
        self.c_cov_y = cov_y
        self.c_cov_sensor_noise = cov_sensor_noise
        self.use_heuristic = use_heuristic
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
        peds = info["peds"]
        car = info["car"]
        is_goal = info["is_goal"]
        is_terminal = info["is_terminal"]
        dist = peds[:, 2:4] - car[2:4]

        # update reward and done bool

        if (is_goal):  # We found a crash
            reward = 0
        elif (is_terminal):
            # reward = 0
            # Heuristic reward based on distance between car and ped at end
            if self.use_heuristic:
                heuristic_reward = np.min(np.linalg.norm(dist, axis=1))
            else:
                # No Herusitic
                heuristic_reward = 0
            reward = -100000 - 10000 * heuristic_reward  # We reached
            # the horizon with no crash
        else:
            reward = -self.mahalanobis_d(action)  # No crash or horizon yet

        return reward

    def mahalanobis_d(self, action):
        """Calculate the Mahalanobis distance [1]_ between the action and the mean action.

        Parameters
        ----------
        action : array_like
            Action taken by the AST solver.

        Returns
        -------
        float
            The Mahalanobis distance between the action and the mean action.

        References
        ----------
        .. [1] Mahalanobis, Prasanta Chandra. "On the generalized distance in statistics." National Institute of
            Science of India, 1936.
            `<http://library.isical.ac.in:8080/jspui/bitstream/10263/6765/1/Vol02_1936_1_Art05-pcm.pdf>`_
        """
        # Mean action is 0
        mean = np.zeros((6 * self.c_num_peds, 1))
        # Assemble the diagonal covariance matrix
        cov = np.zeros((self.c_num_peds, 6))
        cov[:, 0:6] = np.array([self.c_cov_x, self.c_cov_y,
                                self.c_cov_sensor_noise, self.c_cov_sensor_noise,
                                self.c_cov_sensor_noise, self.c_cov_sensor_noise])
        big_cov = np.diagflat(cov)

        # subtract the mean from our actions
        dif = np.copy(action)
        dif[::2] -= mean[0, 0]
        dif[1::2] -= mean[1, 0]

        # calculate the Mahalanobis distance
        dist = np.dot(np.dot(dif.T, np.linalg.inv(big_cov)), dif)

        return np.sqrt(dist)
