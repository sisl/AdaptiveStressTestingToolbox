from mylab.spaces.ast_spaces import ASTSpaces
from gym.spaces.box import Box
import numpy as np

class BenchmarkSpaces(ASTSpaces):
    def __init__(self,
                 amin, # minimum action bound
                 amax, # Maximum action bound
                 T, # Length of a trajectory
                 ):

        assert(len(amin) == len(amax))
        self.amin = amin
        self.amax = amax
        self.T = T
        super().__init__()

    @property
    def action_space(self):
        """
        Returns a Space object
        """

        return Box(low=self.amin, high=self.amax, dtype=np.float32)

    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        return Box(low=self.amin, high=self.amax, dtype=np.float32)
        # return Box(low=np.array([self.amin for i in range(self.T)]), high=np.array([self.amax for i in range(self.T)]),
                   # dtype=np.float32)
