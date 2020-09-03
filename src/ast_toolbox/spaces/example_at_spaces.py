import numpy as np
from numpy.random import rand
import random

from gym.spaces.box import Box

from ast_toolbox.spaces import ASTSpaces

        # sample time [0 T]
        # time = random.uniform(0, T)

        # sample throttle [0 100] (piece-wise constant)
        # throttle = random.uniform(0, 100)

        # sample brake [0 325] (piece-wise constant)
        # brake = random.uniform(0, 325)

        # x = [time, throttle, brake]


class ExampleATSpaces(ASTSpaces):
    def __init__(self,
                 max_actions=4, # Used only if 'use_01_formulation'
                 min_time=0,
                 max_time=30, # 1, # [0, 1] "time left" scheme
                 min_throttle=0,
                 max_throttle=100,
                 min_brake=0,
                 max_brake=325
                 ):

        # Constant hyper-params -- set by user
        # TODO. uniform distribution objects.
        self.c_max_actions = max_actions
        self.c_min_time = min_time
        self.c_max_time = max_time
        self.c_min_throttle = min_throttle
        self.c_max_throttle = max_throttle
        self.c_min_brake = min_brake
        self.c_max_brake = max_brake
        self.use_01_formulation = False # Note, duplicate in "ExampleATSimulator". (testing)

        super().__init__()


    @property
    def action_space(self):
        """
        Returns a Space object
        """
        # TODO. Find better way to do (3,3) with different ranges for each column (using `shape`)

        if self.use_01_formulation:
            return Box(low=0, high=1, shape=(self.c_max_actions, 3), dtype=np.float32)
        else:
            low = np.array([self.c_min_time, self.c_min_throttle, self.c_min_brake])
            high = np.array([self.c_max_time, self.c_max_throttle, self.c_max_brake])
            return Box(low=low, high=high, dtype=np.float32)


    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        low = np.array([0])
        high = np.array([0])
        return Box(low=low, high=high, dtype=np.float32)
