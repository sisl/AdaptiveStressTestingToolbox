"""Class to define the action and observation spaces for an example AV validation task."""
import numpy as np
from gym.spaces.box import Box

from ast_toolbox.spaces import ASTSpaces


class ExampleAVSpaces(ASTSpaces):
    """Class to define the action and observation spaces for an example AV validation task.

    Parameters
    ----------
    num_peds : int, optional
        The number of pedestrians crossing the street.
    max_path_length : int, optional
        Maximum length of a single rollout.
    v_des : float, optional
        The desired velocity, in meters per second,  for the ego vehicle to maintain
    x_accel_low : float, optional
        The minimum x-acceleration of the pedestrian.
    y_accel_low : float, optional
        The minimum y-acceleration of the pedestrian.
    x_accel_high : float, optional
        The maximum x-acceleration of the pedestrian.
    y_accel_high : float, optional
        The maximum y-acceleration of the pedestrian.
    x_boundary_low : float, optional
        The minimum x-position of the pedestrian.
    y_boundary_low : float, optional
        The minimum y-position of the pedestrian.
    x_boundary_high : float, optional
        The maximum x-position of the pedestrian.
    y_boundary_high : float, optional
        The maximum y-position of the pedestrian.
    x_v_low : float, optional
        The minimum x-velocity of the pedestrian.
    y_v_low : float, optional
        The minimum y-velocity of the pedestrian.
    x_v_high : float, optional
        The maximum x-velocity of the pedestrian.
    y_v_high : float, optional
        The maximum y-velocity of the pedestrian.
    car_init_x : float, optional
        The initial x-position of the ego vehicle.
    car_init_y : float, optional
        The initial y-position of the ego vehicle.
    open_loop : bool, optional
        True if the simulation is open-loop, meaning that AST must generate all actions ahead of time, instead
        of being able to output an action in sync with the simulator, getting an observation back before
        the next action is generated. False to get interactive control, which requires that `blackbox_sim_state`
        is also False.
    """

    def __init__(self,
                 num_peds=1,
                 max_path_length=50,
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
                 car_init_x=-35.0,
                 car_init_y=0.0,
                 open_loop=True,
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
        self.open_loop = open_loop
        self.low_start_bounds = [-1.0, -6.0, -1.0, 5.0, 0.0, -6.0, 0.0, 5.0]
        self.high_start_bounds = [1.0, -1.0, 0.0, 9.0, 1.0, -2.0, 1.0, 9.0]
        self.v_start = [1.0, -1.0, 1.0, -1.0]
        super().__init__()

    @property
    def action_space(self):
        """Returns a definition of the action space of the reinforcement learning problem.

        Returns
        -------
        : `gym.spaces.Space <https://gym.openai.com/docs/#spaces>`_
            The action space of the reinforcement learning problem.
        """
        low = np.array([self.c_x_accel_low, self.c_y_accel_low, -3.0, -3.0, -3.0, -3.0])
        high = np.array([self.c_x_accel_high, self.c_y_accel_high, 3.0, 3.0, 3.0, 3.0])

        for i in range(1, self.c_num_peds):
            low = np.hstack((low, np.array([self.c_x_accel_low, self.c_y_accel_low, 0.0, 0.0, 0.0, 0.0])))
            high = np.hstack((high, np.array([self.c_x_accel_high, self.c_y_accel_high, 1.0, 1.0, 1.0, 1.0])))

        return Box(low=low, high=high, dtype=np.float32)

    @property
    def observation_space(self):
        """Returns a definition of the observation space of the reinforcement learning problem.

        Returns
        -------
        : `gym.spaces.Space <https://gym.openai.com/docs/#spaces>`_
            The observation space of the reinforcement learning problem.
        """

        low = np.array([self.c_x_v_low, self.c_y_v_low, self.c_x_boundary_low, self.c_y_boundary_low])
        high = np.array([self.c_x_v_high, self.c_y_v_high, self.c_x_boundary_high, self.c_y_boundary_high])

        for i in range(1, self.c_num_peds):
            low = np.hstack(
                (low, np.array([self.c_x_v_low, self.c_y_v_low, self.c_x_boundary_low, self.c_y_boundary_low])))
            high = np.hstack(
                (high, np.array([self.c_x_v_high, self.c_y_v_high, self.c_x_boundary_high, self.c_y_boundary_high])))

        if self.open_loop:
            low = self.low_start_bounds[:self.c_num_peds * 2]
            low = low + np.ndarray.tolist(0.0 * np.array(self.v_start))[:self.c_num_peds]
            low = low + [0.75 * self.c_v_des]

            high = self.high_start_bounds[:self.c_num_peds * 2]
            high = high + np.ndarray.tolist(2.0 * np.array(self.v_start))[:self.c_num_peds]
            high = high + [1.25 * self.c_v_des]

            if self.c_car_init_x > 0:
                low = low + [0.75 * self.c_car_init_x]
                high = high + [1.25 * self.c_car_init_x]
            else:
                low = low + [1.25 * self.c_car_init_x]
                high = high + [0.75 * self.c_car_init_x]

        return Box(low=np.array(low), high=np.array(high), dtype=np.float32)
