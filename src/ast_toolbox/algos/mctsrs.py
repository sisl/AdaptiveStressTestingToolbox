import ast_toolbox.mcts.AdaptiveStressTesting as AST
import ast_toolbox.mcts.AdaptiveStressTestingRandomSeed as AST_RS
from ast_toolbox.algos.mcts import MCTS


class MCTSRS(MCTS):
    """
    MCTS with Random Seed as action
    """

    def __init__(self,
                 seed=0,
                 rsg_length=1,
                 **kwargs):
        """
        :param seed: the seed used to generate the initial random seed generator.
        :param rsg_length: the length of the state of teh random seed generator. Set it to higher values for extreme large problems.
        :return: No return value.
        """
        self.seed = seed
        self.rsg_length = rsg_length
        super(MCTSRS, self).__init__(**kwargs)

    def init(self):
        ast_params = AST.ASTParams(self.max_path_length, self.log_interval, self.log_tabular, self.log_dir, self.n_itr)
        ast_params.rsg_length = self.rsg_length
        ast_params.init_seed = self.seed
        self.ast = AST_RS.AdaptiveStressTestRS(p=ast_params, env=self.env, top_paths=self.top_paths)
