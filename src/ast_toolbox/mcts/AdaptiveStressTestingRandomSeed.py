import copy

import gym

import ast_toolbox.mcts.RNGWrapper as RNG
from ast_toolbox.mcts.AdaptiveStressTesting import AdaptiveStressTest


class AdaptiveStressTestRS(AdaptiveStressTest):
    """The AST wrapper for MCTS using random seeds as actions.

    Parameters
    ----------
    kwargs :
        Keyword arguments passed to `ast_toolbox.mcts.AdaptiveStressTesting.AdaptiveStressTest`
    """

    def __init__(self, **kwargs):
        super(AdaptiveStressTestRS, self).__init__(**kwargs)
        self.rsg = RNG.RSG(self.params.rsg_length, self.params.init_seed)
        self.initial_rsg = copy.deepcopy(self.rsg)

    def reset_rsg(self):
        """Reset the random seed generator.
        """
        self.rsg = copy.deepcopy(self.initial_rsg)

    def random_action(self):
        """Randomly sample an action for the rollout.

        Returns
        ----------
        action : :py:class:`ast_toolbox.mcts.AdaptiveStressTestingRandomSeed.ASTRSAction`
            The sampled action.
        """
        self.rsg.next()
        return ASTRSAction(action=copy.deepcopy(self.rsg), env=self.env)

    def explore_action(self, s, tree):
        """Randomly sample an action for the exploration.

        Returns
        ----------
        action : :py:class:`ast_toolbox.mcts.AdaptiveStressTestingRandomSeed.ASTRSAction`
            The sampled action.
        """
        self.rsg.next()
        return ASTRSAction(action=copy.deepcopy(self.rsg), env=self.env)


class ASTRSAction:
    """The AST action containing the random seed.

    Parameters
    ----------
    action :
        The random seed.
        env : :py:class:`ast_toolbox.envs.go_explore_ast_env.GoExploreASTEnv`
            The environment.
    """

    def __init__(self, action, env):
        self.env = env
        self.action = action

    def __hash__(self):
        """The redefined hashing method.

        Returns
        ----------
        hash : int
            The hashing result.
        """
        return hash(self.action)

    def __eq__(self, other):
        """The redefined equal method.

        Returns
        ----------
        is_equal : bool
            Whether the two states are equal.
        """
        return self.action == other.action

    def get(self):
        """Get the true action.

        Returns
        ----------
        action :
            The true actions used in the env.
        """
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
