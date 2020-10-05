import numpy as np

from ast_toolbox.mcts.AdaptiveStressTesting import AdaptiveStressTest
from ast_toolbox.mcts.AdaptiveStressTesting import ASTAction


class AdaptiveStressTestBV(AdaptiveStressTest):
    """The AST wrapper for MCTS using the Blind Value exploration [1]_.

    Parameters
    ----------
    kwargs :
        Keyword arguments passed to `ast_toolbox.mcts.AdaptiveStressTesting.AdaptiveStressTest`

    References
    ----------
    .. [1] Couetoux, Adrien, Hassen Doghmen, and Olivier Teytaud. "Improving the exploration in upper confidence trees."
     International Conference on Learning and Intelligent Optimization. Springer, Berlin, Heidelberg, 2012.
    """

    def __init__(self, **kwargs):
        super(AdaptiveStressTestBV, self).__init__(**kwargs)

    def explore_action(self, s, tree):
        """Sample an action for the exploration using Blind Value.

        Parameters
        ----------
        s : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTState`
            The current state.
        tree : dict
            The searching tree.

        Returns
        ----------
        action : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTAction`
            The sampled action.
        """
        s = tree[s]
        A_explored = s.a.keys()
        if len(A_explored) == 0.0:
            return ASTAction(self.env.action_space.sample())
        UCB = self.getUCB(s)
        sigma_known = np.std([float(UCB[a]) for a in s.a.keys()])

        A_pool = []
        dist_pool = []
        center = (self.env.action_space.low + self.env.action_space.high) / 2.0
        for i in range(self.params.M):
            a = self.env.action_space.sample()
            A_pool.append(a)
            dist = self.getDistance(a, center)
            dist_pool.append(dist)
        sigma_pool = np.std(dist_pool)

        rho = sigma_known / sigma_pool

        BV_max = -np.inf
        a_best = None
        for y in A_pool:
            BV = self.getBV(y, rho, A_explored, UCB)
            if BV > BV_max:
                BV_max = BV
                a_best = y
        return ASTAction(a_best)

    def getDistance(self, a, b):
        """Get the (L2) distance between two actions.

        Parameters
        ----------
        a : :py:class:`numpy.ndarry`
            The first action.
        b : :py:class:`numpy.ndarry`
            The second action.

        Returns
        ----------
        distance : float
            The L2 distance between a and b.
        """
        return np.sqrt(np.sum((a - b)**2))

    def getUCB(self, s):
        """Get the upper confidnece bound for the expected return for evary actions that has been explored at the state.

        Parameters
        ----------
        s : :py:class:`ast_toolbox.MCTSdpw.StateNode`
            The state node in the searching tree

        Returns
        ----------
        UCB : dict
            The dictionary containing the upper confidence bound for each explored action in the state node.
        """
        UCB = dict()
        nS = s.n
        for a in s.a.keys():
            UCB[a] = s.a[a].q + self.params.ec * np.sqrt(np.log(nS) / float(s.a[a].n))
        return UCB

    def getBV(self, y, rho, A, UCB):
        """Calculate the Blind Value for the candidate action y

        Parameters
        ----------
        y : :py:class:`numpy.ndarry`
            The candidate action.
        rho : float
            The standard deviation ratio.
        A : list[:py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTAction`]
            The list of the explored AST actions
        UCB : dict
            The dictionary containing the upper confidence bound for each explored action in the state node.

        Returns
        ----------
        BV : float
            The blind value.
        """
        BVs = []
        for a in A:
            BV = rho * self.getDistance(a.action, y) + UCB[a]
            BVs.append(BV)
        return min(BVs)
