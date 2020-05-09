#import base Simulator class
from ast_toolbox.simulators.example_av_simulator import ExampleAVSimulator
#Used for math and debugging
import numpy as np
import pdb
import ast_toolbox.simulators.rss_metrics as rss

#Define the class
class AVRSSSimulator(ExampleAVSimulator):
    """
    Class template for a non-interactive simulator.
    """
    #Accept parameters for defining the behavior of the system under test[SUT]
    def __init__(self,
                 lat_params,
                 long_params,
                 dt = 0.1,
                 alpha = 0.85,
                 beta = 0.005,
                 v_des = 11.17,
                 delta = 4.0,
                 t_headway = 1.5,
                 a_max = 3.0,
                 s_min = 4.0,
                 d_cmf = 2.0,
                 d_max = 9.0,
                 min_dist_x = 2.5,
                 min_dist_y = 1.4,
                 car_init_x = 35.0,
                 car_init_y = 0.0,
                 action_only = True,
                 **kwargs):

        self.lat_params = lat_params
        self.long_params = long_params
        self.car_traj = []
        self.ped_traj = []
        #initialize the AV Simulator
        super().__init__(1, dt, alpha, beta, v_des, delta, t_headway, a_max, s_min, d_cmf, d_max, min_dist_x,
                         min_dist_y, car_init_x, car_init_y, action_only, **kwargs)

    def is_goal(self):
        super_is_goal = super().is_goal()
        # return super_is_goal
        return super_is_goal and self._fraction_proper_response() < 0.9

    def get_reward_info(self):
        dist = self._peds[:, 2:4] - self._car[2:4]
        th_d = np.min(np.linalg.norm(dist, axis=1))
        th_blame = self._fraction_proper_response()
        xx = np.array([th_d, th_blame])
        return {"terminal_heuristic": np.array([th_d, th_blame]),
                "is_goal": self.is_goal(),
                "is_terminal": self._is_terminal}

    def reset(self, s_0):
        self.car_traj = []
        self.ped_traj = []
        return super().reset(s_0)

    def log(self):
        t = self._step * self.c_dt
        # Store the trajectories in a form that rss_metrics likes
        #NOTE: We flip the x and y coordinates in accordance with RSS conventions
        self.car_traj.append(rss.CarState(t, self._car[3], self._car[2], self._car[1], self._car[0], self._car_accel[1],
                                          self._car_accel[0]))
        ped_a = self._action.reshape((1, 6))[0, 0:2]
        self.ped_traj.append(rss.CarState(t,self._peds[0,3], self._peds[0,2], self._peds[0,1], self._peds[0,0],
                                          ped_a[1], ped_a[0]))

        super().log()

    def move_car(self, car, accel):
        car[2:4] += self.c_dt * car[0:2]
        car[0:2] += self.c_dt * accel
        car[0] = max(0, car[0])
        return car

    def _fraction_proper_response(self):
        # If we dont have enough data points then assume the response is proper
        if np.size(self.car_traj) == 0:
            return 1
        car_resp, _, _, _ = rss.characterize_response(self.car_traj, self.ped_traj, self.lat_params, self.long_params)
        return np.count_nonzero(car_resp == rss.Response.Proper) / np.size(car_resp)