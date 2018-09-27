# import base class
from mylab.rewards.ast_reward import ASTReward

# useful packages for math and debugging
import numpy as np
import pdb

# Define the class, inherit from the base
class ExampleAVReward(ASTReward):
    def __init__(self,
                 num_peds=1,
                 cov_x=0.1,
                 cov_y=0.01,
                 cov_sensor_noise=0.1):

        self.c_num_peds = num_peds
        self.c_cov_x = cov_x
        self.c_cov_y = cov_y
        self.c_cov_sensor_noise = cov_sensor_noise
        super().__init__()

    def give_reward(self, action, **kwargs):
        # get the info from the simulator
        info = kwargs['info']
        peds = info["peds"]
        car = info["car"]
        is_goal = info["is_goal"]
        is_terminal = info["is_terminal"]
        dist = peds[:, 2:4] - car[2:4]

        # update reward and done bool

        if (is_goal): # We found a crash
            reward = 0
        elif (is_terminal):
            reward = -10000 - 1000 * np.min(np.linalg.norm(dist, axis=1)) # We reached
            # the horizon with no crash
        else:
            reward = -np.log(1 + self.mahalanobis_d(action)) # No crash or horizon yet

        return reward


    def mahalanobis_d(self, action):
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