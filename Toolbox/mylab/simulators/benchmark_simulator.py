#import base Simulator class
from mylab.simulators.ast_simulator import ASTSimulator
#Used for math and debugging
import numpy as np
import pdb
#Define the class
class BenchmarkSimulator(ASTSimulator):
    """
    Class template for a non-interactive simulator.
    """
    #Accept parameters for defining the behavior of the system under test[SUT]
    def __init__(self,
                 target_action_sequences, # set of action sequences that represent to events of interest
                 distance_measure, # Measure of the distance between two action sequences
                 tolerance_for_match, # difference between trajectories below which they are considered equal
                 **kwargs):

        self.tolerance = tolerance_for_match
        self.D = distance_measure
        self.E = target_action_sequences
        self.action_sequence = np.zeros((0, self.E[0].shape[1]))
        self._info = []

        #initialize the base Simulator
        super().__init__(**kwargs)



    def simulate(self, actions, s_0):
        print("simulate!!!!")
        self.action_sequence = actions
        self.log()

    def step(self, action, open_loop):
        self.action_sequence = np.vstack([self.action_sequence, action])
        if self.action_sequence.shape[0] >= self.c_max_path_length:
            self._is_terminal = True

        return action

    def reset(self, s_0):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._is_terminal = False
        self.action_sequence = s_0
        return np.zeros(2)


    def _min_distance(self):
        return min([self.D(e, self.action_sequence) for e in self.E])

    def get_reward_info(self):
        """
        returns any info needed by the reward function to calculate the current reward
        """
        return {"terminal_heuristic": self._min_distance(),
                "is_goal": self.is_goal(),
                "is_terminal": self._is_terminal,
                "action_sequence": self.action_sequence}

    def is_goal(self):
        """
        returns whether the current state is in the goal set
        :return: boolean, true if current state is in goal set.
        """
        return self._min_distance() < self.tolerance

    def log(self):
        print('skipping log step')
        # Create a cache of step specific variables for post-simulation analysis
        # cache = np.hstack([0.0,  # Dummy, will be filled in with trial # during post processing in save_trials.py
        #                    self._step,
        #                    np.ndarray.flatten(self._car),
        #                    np.ndarray.flatten(self._peds),
        #                    np.ndarray.flatten(self._action),
        #                    0.0])
        # self._info.append(cache)
        # self._step += 1

