import ast_toolbox.mcts.AdaptiveStressTesting as AST
import ast_toolbox.mcts.AdaptiveStressTestingBlindValue as AST_BV
from ast_toolbox.algos.mcts import MCTS


class MCTSBV(MCTS):
    """Monte Carlo Tress Search (MCTS) with double progressive widening (DPW) [1]_ using Blind Value search from Couetoux et al.  [2]_.

    Parameters
    ----------
    M : int, optional
        The number of randon decisions generated for the action pool.
    kwargs :
        Keyword arguments passed to `ast_toolbox.algos.mcts.MCTS`.

    References
    ----------
    .. [1] Lee, Ritchie, et al. "Adaptive stress testing of airborne collision avoidance systems."
     2015 IEEE/AIAA 34th Digital Avionics Systems Conference (DASC). IEEE, 2015.
    .. [2] Couetoux, Adrien, Hassen Doghmen, and Olivier Teytaud. "Improving the exploration in upper confidence trees."
     International Conference on Learning and Intelligent Optimization. Springer, Berlin, Heidelberg, 2012.
    """

    def __init__(self,
                 M=10,
                 **kwargs):
        self.ec = kwargs['ec']
        self.M = M
        super(MCTSBV, self).__init__(**kwargs)

    def init(self):
        """Initiate AST internal parameters
        """
        ast_params = AST.ASTParams(self.max_path_length, self.log_interval, self.log_tabular, self.log_dir, self.n_itr)
        ast_params.ec = self.ec
        ast_params.M = self.M
        self.ast = AST_BV.AdaptiveStressTestBV(p=ast_params, env=self.env, top_paths=self.top_paths)
