import copy

import gym

import ast_toolbox.mcts.RNGWrapper as RNG
from ast_toolbox.mcts.AdaptiveStressTesting import AdaptiveStressTest


class AdaptiveStressTestRS(AdaptiveStressTest):
    def __init__(self, **kwargs):
        super(AdaptiveStressTestRS, self).__init__(**kwargs)
        self.rsg = RNG.RSG(self.params.rsg_length, self.params.init_seed)
        self.initial_rsg = copy.deepcopy(self.rsg)

    def reset_rsg(self):
        self.rsg = copy.deepcopy(self.initial_rsg)

    def random_action(self):
        self.rsg.next()
        return ASTRSAction(action=copy.deepcopy(self.rsg), env=self.env)

    def explore_action(self, s, tree):
        self.rsg.next()
        return ASTRSAction(action=copy.deepcopy(self.rsg), env=self.env)


class ASTRSAction:
    def __init__(self, action, env):
        self.env = env
        self.action = action

    def __hash__(self):
        return hash(self.action)

    def __eq__(self, other):
        return self.action == other.action

    def get(self):
        rng_state = self.action.state
        # TODO: a better approch to make use of random seed of length > 1
        action_seed = int(rng_state[0])
        if isinstance(self.env.action_space, gym.spaces.Space):
            action_space = self.env.action_space
            # need to do this since every time call env.action_space, a new space is created
            action_space.seed(action_seed)
            true_action = action_space.sample()
        else:
            true_action = action_seed
        return true_action
