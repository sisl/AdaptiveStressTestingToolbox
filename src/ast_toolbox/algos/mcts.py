import numpy as np

import ast_toolbox.mcts.AdaptiveStressTesting as AST
import ast_toolbox.mcts.AST_MCTS as AST_MCTS
import ast_toolbox.mcts.ASTSim as ASTSim
import ast_toolbox.mcts.MCTSdpw as MCTSdpw
from ast_toolbox.mcts import tree_plot


class MCTS:
    """Monte Carlo Tress Search (MCTS) with double progressive widening (DPW) [1]_ using the env's action space as its action space.

    Parameters
    ----------
    env : :py:class:`ast_toolbox.envs.go_explore_ast_env.GoExploreASTEnv`.
        The environment.
    max_path_length: int
        The maximum search depth.
    ec : float
        The exploration constant used in UCT equation.
    n_itr: int
            The iteration number, the total numeber of environment call is approximately n_itr*max_path_length*max_path_length.
    k : float
        The constraint parameter used in DPW: |N(s,a)|<=kN(s)^alpha.
    alpha : float
        The constraint parameter used in DPW: |N(s,a)|<=kN(s)^alpha.
    clear_nodes : bool
        Whether to clear redundant nodes in tree.
        Set it to True for saving memoray. Set it to False to better tree plotting.
    log_interval : int
        The log interval in terms of environment calls.
    top_paths : :py:class:`ast_toolbox.mcts.BoundedPriorityQueues`, optional
        The bounded priority queue to store top-rewarded trajectories.
    gamma : float, optional
        The discount factor.
    stress_test_mode : int, optional
        The mode of the tree search. 1 for single tree. 2 for multiple trees.
    log_tabular : bool, optional
        Whether to log the training statistics into a tabular file.
    plot_tree : bool, optional
        Whether to plot the resulting searching tree.
    plot_path : str, optional
        The storing path for the tree plot.
    plot_format : str, optional
        The storing format for the tree plot

    References
    ----------
    .. [1] Lee, Ritchie, et al. "Adaptive stress testing of airborne collision avoidance systems."
     2015 IEEE/AIAA 34th Digital Avionics Systems Conference (DASC). IEEE, 2015.
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
        """Initiate AST internal parameters
        """
        ast_params = AST.ASTParams(self.max_path_length, self.log_interval, self.log_tabular, self.log_dir, self.n_itr)
        self.ast = AST.AdaptiveStressTest(p=ast_params, env=self.env, top_paths=self.top_paths)

    def train(self, runner):
        """Start training.

        Parameters
        ----------
        runner : :py:class:`garage.experiment.LocalRunner <garage:garage.experiment.LocalRunner>`
            ``LocalRunner`` is passed to give algorithm the access to ``runner.step_epochs()``, which provides services
            such as snapshotting and sampler control.
        """
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
