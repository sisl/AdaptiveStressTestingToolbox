import ast_toolbox.mcts.AdaptiveStressTesting as AST
import ast_toolbox.mcts.AdaptiveStressTestingRandomSeed as AST_RS
from ast_toolbox.algos.mcts import MCTS


class MCTSRS(MCTS):
    """Monte Carlo Tress Search (MCTS) with double progressive widening (DPW) [1]_ using the random seeds as its action space.

    Parameters
    ----------
    seed : int, optional
        The seed used to generate the initial random seed generator.
    rsg_length: int, optional
        The length of the state of the random seed generator. Set it to higher values for extreme large problems.

    References
    ----------
    .. [1] Lee, Ritchie, et al. "Adaptive stress testing of airborne collision avoidance systems."
     2015 IEEE/AIAA 34th Digital Avionics Systems Conference (DASC). IEEE, 2015.
    """

    def __init__(self,
                 seed=0,
                 rsg_length=1,
                 **kwargs):
        self.seed = seed
        self.rsg_length = rsg_length
        super(MCTSRS, self).__init__(**kwargs)

    def init(self):
        """Initiate AST internal parameters
        """
        ast_params = AST.ASTParams(self.max_path_length, self.log_interval, self.log_tabular, self.log_dir, self.n_itr)
        ast_params.rsg_length = self.rsg_length
        ast_params.init_seed = self.seed
        self.ast = AST_RS.AdaptiveStressTestRS(p=ast_params, env=self.env, top_paths=self.top_paths)
