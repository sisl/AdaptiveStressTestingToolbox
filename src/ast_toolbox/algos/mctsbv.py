import ast_toolbox.mcts.AdaptiveStressTesting as AST
import ast_toolbox.mcts.AdaptiveStressTestingBlindValue as AST_BV
from ast_toolbox.algos.mcts import MCTS


class MCTSBV(MCTS):
    """
    MCTS with Blind Value
    """

    def __init__(self,
                 M=10,
                 **kwargs):
        """
        :param M: the number of randon decisions generated for the action pool
        :return: No return value.
        """
        self.ec = kwargs['ec']
        self.M = M
        super(MCTSBV, self).__init__(**kwargs)

    def init(self):
        ast_params = AST.ASTParams(self.max_path_length, self.log_interval, self.log_tabular, self.log_dir, self.n_itr)
        ast_params.ec = self.ec
        ast_params.M = self.M
        self.ast = AST_BV.AdaptiveStressTestBV(p=ast_params, env=self.env, top_paths=self.top_paths)
