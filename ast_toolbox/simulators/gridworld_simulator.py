# from garage.envs.base import GarageEnv
# from garage.envs.base import Step
# from garage.spaces import Box
from ast_toolbox.simulators.ast_simulator import ASTSimulator
import numpy as np

import pdb

class GridworldSimulator(ASTSimulator):
    """
    Simulate a gridworld scenario
    """
    def __init__(self,
                 goal_list,
                 grid_mins,
                 grid_maxes,
                 **kwargs):
        """
        :function goal_set - function definition that accepts a state, and returns true if state is in set.
        :parameter  s_0 - the initial state of the simulator.
        """
        grid_lengths = np.abs(grid_mins) + np.abs(grid_maxes) + 1
        self.goals = np.concatenate(np.unravel_index(goal_list, grid_lengths)).reshape((len(goal_list), -1)).T


        super().__init__(**kwargs)


    def simulate(self, actions, s_0):
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
        path_length = 0
        self.reset(s_0)
        self._info = []

        # Take simulation steps unbtil horizon is reached
        while path_length < self.c_max_path_length:
            self.state += actions[path_length]

            # check if a crash has occurred. If so return the timestep, otherwise continue
            if self.is_goal():
                return path_length, np.array(self._info)
            path_length = path_length + 1

        self._is_terminal = True
        return -1, np.array(self._info)

    def closed_loop_step(self, action):
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
        self.state += action
        self.observation = self.state

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
        self.state = s_0
        self.initial_conditions = s_0

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

    def is_terminal(self):
        return self._is_terminal

    def log(self):
        """
        perform any logging steps
        """
        pass

    def clone_state(self):
        pass

    def restore_state(self, in_simulator_state):
        simulator_state = in_simulator_state.copy()
