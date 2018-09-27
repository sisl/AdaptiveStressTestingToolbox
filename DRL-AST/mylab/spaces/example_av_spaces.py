from mylab.spaces.ast_spaces import ASTSpaces
from rllab.spaces import Box
import numpy as np

class ExampleAVSpaces(ASTSpaces):
    def __init__(self,
                 num_peds=1,
                 max_path_length = 50,
                 v_des=11.17,
                 x_accel_low=-1.0,
                 y_accel_low=-1.0,
                 x_accel_high=1.0,
                 y_accel_high=1.0,
                 x_boundary_low=-10.0,
                 y_boundary_low=-10.0,
                 x_boundary_high=10.0,
                 y_boundary_high=10.0,
                 x_v_low=-10.0,
                 y_v_low=-10.0,
                 x_v_high=10.0,
                 y_v_high=10.0,
                 car_init_x=35.0,
                 car_init_y=0.0,
                 action_only = True,
                 ):

        # Constant hyper-params -- set by user
        self.c_num_peds = num_peds
        self.c_max_path_length = max_path_length
        self.c_v_des = v_des
        self.c_x_accel_low = x_accel_low
        self.c_y_accel_low = y_accel_low
        self.c_x_accel_high = x_accel_high
        self.c_y_accel_high = y_accel_high
        self.c_x_boundary_low = x_boundary_low
        self.c_y_boundary_low = y_boundary_low
        self.c_x_boundary_high = x_boundary_high
        self.c_y_boundary_high = y_boundary_high
        self.c_x_v_low = x_v_low
        self.c_y_v_low = y_v_low
        self.c_x_v_high = x_v_high
        self.c_y_v_high = y_v_high
        self.c_car_init_x = car_init_x
        self.c_car_init_y = car_init_y
        self.action_only = action_only
        self.low_start_bounds = [-1.0, -4.25, -1.0, 5.0, 0.0, -6.0, 0.0, 5.0]
        self.high_start_bounds = [0.0, -3.75, 0.0, 9.0, 1.0, -2.0, 1.0, 9.0]
        self.v_start = [1.0, -1.0, 1.0, -1.0]
        super().__init__()

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        low = np.array([self.c_x_accel_low, self.c_y_accel_low, 0.0, 0.0, 0.0, 0.0])
        high = np.array([self.c_x_accel_high, self.c_y_accel_high, 1.0, 1.0, 1.0, 1.0])

        for i in range(1, self.c_num_peds):
            low = np.hstack((low, np.array([self.c_x_accel_low, self.c_y_accel_low, 0.0, 0.0, 0.0, 0.0])))
            high = np.hstack((high, np.array([self.c_x_accel_high, self.c_y_accel_high, 1.0, 1.0, 1.0, 1.0])))

        return Box(low=low, high=high)

    @property
    def observation_space(self):
        """
        Returns a Space object
        """

        low = np.array([self.c_x_v_low, self.c_y_v_low, self.c_x_boundary_low, self.c_y_boundary_low])
        high = np.array([self.c_x_v_high, self.c_y_v_high, self.c_x_boundary_high, self.c_y_boundary_high])

        for i in range(1, self.c_num_peds):
            low = np.hstack(
                (low, np.array([self.c_x_v_low, self.c_y_v_low, self.c_x_boundary_low, self.c_y_boundary_low])))
            high = np.hstack(
                (high, np.array([self.c_x_v_high, self.c_y_v_high, self.c_x_boundary_high, self.c_y_boundary_high])))

        if self.action_only:
            low = self.low_start_bounds[:self.c_num_peds * 2]
            low = low + np.ndarray.tolist(0.0 * np.array(self.v_start))[:self.c_num_peds]
            low = low + [0.75 * self.c_v_des]
            low = low + [0.75 * self.c_car_init_x]
            high = self.high_start_bounds[:self.c_num_peds * 2]
            high = high + np.ndarray.tolist(2.0 * np.array(self.v_start))[:self.c_num_peds]
            high = high + [1.25 * self.c_v_des]
            high = high + [1.25 * self.c_car_init_x]

        # pdb.set_trace()
        return Box(low=np.array(low), high=np.array(high))
