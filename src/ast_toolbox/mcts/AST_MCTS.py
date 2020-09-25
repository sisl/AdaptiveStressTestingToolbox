import ast_toolbox.mcts.MCTSdpw as MCTSdpw
import ast_toolbox.mcts.MDP as MDP


def rollout_getAction(ast):
    """Get the rollout function from ast.

    Parameters
    ----------
    ast : :py:class:`ast_toolbox.mcts.AdaptiveStressTest.AdaptiveStressTesting`
        The AST object.
    """
    def rollout_policy(s, tree):
        return ast.random_action()
    return rollout_policy


def explore_getAction(ast):
    """Get the exploration function from ast.

    Parameters
    ----------
    ast : :py:class:`ast_toolbox.mcts.AdaptiveStressTest.AdaptiveStressTesting`
        The AST object.
    """
    def explore_policy(s, tree):
        return ast.explore_action(s, tree)
    return explore_policy


def stress_test(ast, mcts_params, top_paths, verbose=True, return_tree=False):
    """Run stress test with mode 1 (search with single tree).

    Parameters
    ----------
    ast : :py:class:`ast_toolbox.mcts.AdaptiveStressTest.AdaptiveStressTesting`
        The AST object.
    mcts_params: :py:class:`ast_toolbox.mcts.MCTSdpw.DPWParams`
        The mcts parameters.
    top_paths : :py:class:`ast_toolbox.mcts.BoundedPriorityQueues`
        The bounded priority queue to store top-rewarded trajectories.
    verbose : bool, optional
        Whether to logging test information
    return_tree: bool, optional
        Whether to return the search tree

    Returns
    -------
    results : :py:class:`ast_toolbox.mcts.AdaptiveStressTest.AdaptiveStressTesting`
        The bounded priority queue storing top-rewarded trajectories.
    tree : dict
        The resulting searching tree.
    """
    dpw_model = MCTSdpw.DPWModel(ast.transition_model, rollout_getAction(ast), explore_getAction(ast))
    tree = MCTSdpw.DPWTree(mcts_params, dpw_model)
    (mcts_reward, action_seq) = MDP.simulate(tree.f.model, tree, MCTSdpw.selectAction, verbose=verbose)

    results = ast.top_paths
    if return_tree:
        return results, tree.s_tree
    else:
        return results


def stress_test2(ast, mcts_params, top_paths, verbose=True, return_tree=False):
    """Run stress test with mode 2 (search with multiple trees).

    Parameters
    ----------
    ast : :py:class:`ast_toolbox.mcts.AdaptiveStressTest.AdaptiveStressTesting`
        The AST object.
    mcts_params: :py:class:`ast_toolbox.mcts.MCTSdpw.DPWParams`
        The mcts parameters.
    top_paths : :py:class:`ast_toolbox.mcts.BoundedPriorityQueues`
        The bounded priority queue to store top-rewarded trajectories.
    verbose : bool, optional
        Whether to logging test information
    return_tree: bool, optional
        Whether to return the search tree

    Returns
    -------
    results : :py:class:`ast_toolbox.mcts.AdaptiveStressTest.AdaptiveStressTesting`
        The bounded priority queue storing top-rewarded trajectories.
    tree : dict
        The resulting searching tree.
    """
    mcts_params.clear_nodes = False
    mcts_params.n *= ast.params.max_steps

    dpw_model = MCTSdpw.DPWModel(ast.transition_model, rollout_getAction(ast), explore_getAction(ast))
    tree = MCTSdpw.DPWTree(mcts_params, dpw_model)

    s = tree.f.model.getInitialState()
    MCTSdpw.selectAction(tree, s, verbose=verbose)

    results = ast.top_paths
    if return_tree:
        return results, tree.s_tree
    else:
        return results
