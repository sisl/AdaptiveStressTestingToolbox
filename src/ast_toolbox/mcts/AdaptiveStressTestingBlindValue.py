import numpy as np

from ast_toolbox.mcts.AdaptiveStressTesting import AdaptiveStressTest
from ast_toolbox.mcts.AdaptiveStressTesting import ASTAction


class AdaptiveStressTestBV(AdaptiveStressTest):
    def __init__(self, **kwargs):
        super(AdaptiveStressTestBV, self).__init__(**kwargs)

    def explore_action(self, s, tree):
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
        return np.sqrt(np.sum((a - b)**2))

    def getUCB(self, s):
        UCB = dict()
        nS = s.n
        for a in s.a.keys():
            UCB[a] = s.a[a].q + self.params.ec * np.sqrt(np.log(nS) / float(s.a[a].n))
        return UCB

    def getBV(self, y, rho, A, UCB):
        BVs = []
        for a in A:
            BV = rho * self.getDistance(a.action, y) + UCB[a]
            BVs.append(BV)
        return min(BVs)
