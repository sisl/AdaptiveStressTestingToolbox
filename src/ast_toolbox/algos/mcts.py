import numpy as np

import ast_toolbox.mcts.AdaptiveStressTesting as AST
import ast_toolbox.mcts.AST_MCTS as AST_MCTS
import ast_toolbox.mcts.ASTSim as ASTSim
import ast_toolbox.mcts.MCTSdpw as MCTSdpw
from ast_toolbox.mcts import tree_plot


class MCTS:
    """
    MCTS
    """

    def __init__(
        self,
        env,
        max_path_length,
        ec,
        n_itr,
        k,
        alpha,
        clear_nodes,
        log_interval,
        top_paths,
        log_dir,
        gamma=1.0,
        stress_test_mode=2,
        log_tabular=True,
        plot_tree=False,
        plot_path=None,
        plot_format='png'
    ):
        """
        :param env: the task environment
        :param max_path_length: maximum search depth
        :param ec: exploration constant used in UCT equation
        :param n_itr: iteration number, the total numeber of environment call is approximately
                                        n_itr*max_path_length*max_path_length
        :param k, alpha: the constraint parameter used in DPW: |N(s,a)|<=kN(s)^alpha
        :param clear_nodes: whether to clear redundant nodes in tree.
                                        Set it to True for saving memoray. Set it to False to better tree plotting
        :param log_interval: the log interval in terms of environment calls
        :param top_paths: a bounded priority queue to store top-rewarded trajectories
        :param gamma: discount factor
        :param plot_tree, plot_path, plot_format: tree plotting parameters
        :return: No return value.
        """
        self.env = env
        self.stress_test_mode = stress_test_mode
        self.max_path_length = max_path_length
        self.macts_params = MCTSdpw.DPWParams(max_path_length,
                                              gamma,
                                              ec,
                                              2 * max(n_itr * log_interval // max_path_length ** 2, 1),
                                              k,
                                              alpha,
                                              clear_nodes)
        self.log_interval = log_interval
        self.top_paths = top_paths
        self.log_tabular = log_tabular
        self.plot_tree = plot_tree
        self.plot_path = plot_path
        self.plot_format = plot_format
        self.policy = None
        self.log_dir = log_dir
        self.n_itr = n_itr

    def init(self):
        ast_params = AST.ASTParams(self.max_path_length, self.log_interval, self.log_tabular, self.log_dir, self.n_itr)
        self.ast = AST.AdaptiveStressTest(p=ast_params, env=self.env, top_paths=self.top_paths)

    def train(self, runner):
        self.init()
        if self.plot_tree:
            if self.stress_test_mode == 2:
                result, tree = AST_MCTS.stress_test2(self.ast, self.macts_params, self.top_paths, verbose=False, return_tree=True)
            else:
                result, tree = AST_MCTS.stress_test(self.ast, self.macts_params, self.top_paths, verbose=False, return_tree=True)
        else:
            if self.stress_test_mode == 2:
                result = AST_MCTS.stress_test2(self.ast, self.macts_params, self.top_paths, verbose=False, return_tree=False)
            else:
                result = AST_MCTS.stress_test(self.ast, self.macts_params, self.top_paths, verbose=False, return_tree=False)
        self.ast.params.log_tabular = False
        print("checking reward consistance")
        for (action_seq, reward_predict) in result:
            [a.get() for a in action_seq]
            reward, _ = ASTSim.play_sequence(self.ast, action_seq, sleeptime=0.0)
            assert np.isclose(reward_predict, reward)
        print("done")
        if self.plot_tree:
            tree_plot.plot_tree(tree, d=self.max_path_length, path=self.plot_path, format=self.plot_format)
