
# useful packages for math and debugging
import numpy as np
import pdb
from mylab.rewards.action_model import ActionModel
from scipy.stats import multivariate_normal

# Define the class, inherit from the base
class PedestrianNoiseGaussian(ActionModel):
    def __init__(self,
                 num_peds=1,
                 cov_x=0.1,
                 cov_y=0.01,
                 cov_sensor_noise=0.1):

        self.c_num_peds = num_peds
        self.c_cov_x = cov_x
        self.c_cov_y = cov_y
        self.c_cov_sensor_noise = cov_sensor_noise

    def log_prob(self, action):
        mean = np.zeros(6 * self.c_num_peds)
        cov = np.zeros((self.c_num_peds, 6))
        cov[:, 0:6] = np.array([self.c_cov_x, self.c_cov_y,
                                self.c_cov_sensor_noise, self.c_cov_sensor_noise,
                                self.c_cov_sensor_noise, self.c_cov_sensor_noise])
        big_cov = np.diagflat(cov)
        pdf = multivariate_normal.pdf(action, mean=mean, cov=big_cov)
        logpdf = max(np.log(pdf), -100)
        return logpdf