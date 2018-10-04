"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
from Cartpole.cartpole import CartPoleEnv
from rllab.misc.overrides import overrides
import numpy as np
import rllab.spaces as spaces

class CartPoleNdEnv(CartPoleEnv):
    def __init__(self, nd ,*args, **kwargs):
        self.nd = nd
        super(CartPoleNdEnv, self).__init__(*args, **kwargs)

    @property
    def ast_action_space(self):
        high = np.array([self.wind_force_mag for i in range(self.nd)])
        return spaces.Box(-high,high)

    def ast_step(self, action, ast_action):
        if self.use_seed:
            np.random.seed(ast_action)
            ast_action = self.ast_action_space.sample()
        ast_action = np.mean(ast_action)
        use_seed = self.use_seed
        self.use_seed = False
        results = super(CartPoleNdEnv, self).ast_step(action, ast_action)
        self.use_seed = use_seed
        return results