"""Class to define the action and observation spaces for an example AV validation task."""
import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.tuple import Tuple

from ast_toolbox.spaces import ASTSpaces


class CrazyTrolleySpaces(ASTSpaces):
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
                 max_seed=1e6,
                 ):
        # Constant hyper-params -- set by user
        self.max_seed = max_seed
        super().__init__()

    @property
    def action_space(self):
        """Returns a definition of the action space of the reinforcement learning problem.

        Returns
        -------
        : `gym.spaces.Space <https://gym.openai.com/docs/#spaces>`_
            The action space of the reinforcement learning problem.
        """

        return Tuple([Discrete(self.max_seed)])

    @property
    def observation_space(self):
        """Returns a definition of the observation space of the reinforcement learning problem.

        Returns
        -------
        : `gym.spaces.Space <https://gym.openai.com/docs/#spaces>`_
            The observation space of the reinforcement learning problem.
        """

        return Tuple([Discrete(self.max_seed)])
