from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np

import pdb

class Simulator(object):
    """
    Class template for a non-interactive simulator.
    """
    def __init__(self, max_path_length = 50):
        """
        :function goal_set - function definition that accepts a state, and returns true if state is in set.
        :parameter  s_0 - the initial state of the simulator.
        """
        self.c_max_path_length = max_path_length
        self._is_terminal = False


    def simulate(self, action, s_0):
        """
        Run/finish the simulation
        Input
        -----
        action : A sequential list of actions taken by the simulation
        Outputs
        -------
        (terminal_index)
        terminal_index : The index of the action that resulted in a state in the goal set E. If no state is found
                        terminal_index should be returned as -1.

        """
        raise NotImplementedError

    def step(self, action):
        """
        Handle anything that needs to take place at each step, such as a simulation update or write to file
        Input
        -----
        action : action taken on the turn
        Outputs
        -------
        (terminal_index)
        terminal_index : The index of the action that resulted in a state in the goal set E. If no state is found
                        terminal_index should be returned as -1.

        """
        raise NotImplementedError

    def reset(self, s_0):
        """
        Resets the state of the environment, returning an initial observation.
        Inputs
        -------
        s_0: the initial state
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        raise NotImplementedError

    def get_reward_info(self):
        """
        returns any info needed by the reward function to calculate the current reward
        """
        raise NotImplementedError

    def is_goal(self):
        """
        returns whether the current state is in the goal set
        :return: boolean, true if current state is in goal set.
        """
        raise NotImplementedError

    def log(self):
        """
        perform any logging steps
        """
        pass

